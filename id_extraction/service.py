"""
Business logic for Australian ID extraction.

This module is deliberately framework-independent: it knows nothing about
Flask or Streamlit. Both entry points call the same three functions, so
the rules live in exactly one place:

    streamlit_app.py  --(in-process import)-->  service  <--(HTTP)--  api.py

Every response is already display-ready. "status" is a Streamlit method
name ("success" | "warning" | "error"), "message" and "icon" are what to
show the user, and "fields" is the ordered list to render as a form. The
UI makes no decisions about any of it -- it only renders what it is given.
"""

import os
import re
import json
import time
import uuid
import tempfile
import datetime

import cv2

from .ocr_processors import (
    MedicareProcessor,
    PassportProcessor,
    DrivingLicenceProcessor,
    build_ocr_kwargs,
    check_image_resolution,
    OCR_DEVICE,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where reviewed + confirmed records are appended. NOTE: on Streamlit
# Community Cloud the filesystem is EPHEMERAL -- this file disappears when
# the app restarts or redeploys. Point SUBMISSIONS_FILE at a mounted volume,
# or replace _persist_record() with a real database, for durable storage.
SUBMISSIONS_FILE = os.environ.get("SUBMISSIONS_FILE", "submissions.jsonl")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))


# ---------------------------------------------------------------------------
# Document definitions: everything the UI needs to know lives here, so the
# frontend has no document-specific knowledge of its own.
# ---------------------------------------------------------------------------

FIELD_SCHEMAS = {
    "medicare": [
        ("Medicare Number", "Medicare Number:"),
        ("Valid To", "Valid To:"),
        ("Cardholder 1", "Cardholder 1:"),
        ("Cardholder 2", "Cardholder 2:"),
        ("Cardholder 3", "Cardholder 3:"),
        ("Cardholder 4", "Cardholder 4:"),
        ("Cardholder 5", "Cardholder 5:"),
    ],
    "passport": [
        ("Surname", "Surname:"),
        ("Given Names", "Given Names:"),
        ("Passport Number", "Passport Number:"),
        ("Nationality", "Nationality:"),
        ("Date of Birth", "Date of Birth:"),
        ("Sex", "Sex:"),
        ("Place of Birth", "Place of Birth:"),
        ("Date of Issue", "Date of Issue:"),
        ("Date of Expiry", "Date of Expiry:"),
    ],
    "licence": [
        ("Name", "Name:"),
        ("Address", "Address:"),
        ("Licence Number", "Licence Number:"),
        ("Licence Class", "Licence Class:"),
        ("Conditions", "Conditions:"),
        ("Donor", "Donor:"),
        ("Date of Birth", "Date of Birth:"),
        ("Expiry Date", "Expiry Date:"),
        ("Card Number", "Card Number:"),
        ("Valid In", "Valid In:"),
    ],
}

DOC_TYPES = [
    {"key": "medicare", "label": "Medicare Card", "endpoint": "/extract/medicare"},
    {"key": "passport", "label": "Passport", "endpoint": "/extract/passport"},
    {"key": "licence", "label": "Driving Licence", "endpoint": "/extract/licence"},
]

REQUIRED_FIELDS = {
    "medicare": ["Medicare Number"],
    "passport": ["Passport Number", "Surname", "Given Names", "Date of Expiry"],
    "licence": ["Licence Number", "Name", "Expiry Date"],
}

# Medicare's "Valid To" is printed as MM/YY (2 parts), unlike the 3-part
# dates on a passport or licence -- validating it with the 3-part pattern
# would reject a correctly-extracted value.
FULL_DATE_FIELDS = {
    "passport": ["Date of Birth", "Date of Issue", "Date of Expiry"],
    "licence": ["Date of Birth", "Expiry Date"],
}
MONTH_YEAR_FIELDS = {
    "medicare": ["Valid To"],
}

# Accepts the numeric form (24/3/1937) and the spelled-month form that
# Australian licences and passports actually print (01 JAN 2000).
_FULL_DATE_RE = re.compile(
    r"^\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}$"
    r"|^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}$"
)
_MONTH_YEAR_RE = re.compile(r"^\d{1,2}[\/\.\-]\d{2,4}$")
_MEDICARE_NO_RE = re.compile(r"^\d{4}\s?\d{5}\s?\d$")


# ---------------------------------------------------------------------------
# Lazy OCR engine. Models are loaded on first use, not at import time, so
# importing this module stays cheap (matters for tests and for the Flask
# process starting up).
# ---------------------------------------------------------------------------

_processors = None


def get_processors():
    global _processors
    if _processors is None:
        from paddleocr import PaddleOCR
        engine = PaddleOCR(**build_ocr_kwargs())
        _processors = {
            "medicare": MedicareProcessor(ocr_engine=engine),
            "passport": PassportProcessor(ocr_engine=engine),
            "licence": DrivingLicenceProcessor(ocr_engine=engine),
        }
    return _processors


def ocr_device():
    return OCR_DEVICE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(message, icon="❌"):
    return {"status": "error", "icon": icon, "message": message}


def allowed_file(filename):
    return (
        "." in (filename or "")
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _build_fields(doc_type, extracted):
    """Turn the processor's raw dict into the ordered list the UI renders."""
    return [
        {"key": key, "label": label, "value": extracted.get(key)}
        for key, label in FIELD_SCHEMAS.get(doc_type, [])
    ]


def _blurry_message(blur_score):
    msg = "Image is too blurry to be processed reliably."
    if blur_score is not None:
        msg += f"\n\nBlur score: {blur_score}"
    msg += (
        "\n\nPlease upload a clearer image:\n"
        "- Better lighting\n"
        "- No motion blur\n"
        "- Card/document fully visible and flat"
    )
    return msg


# ---------------------------------------------------------------------------
# Public API -- each returns (payload, http_status)
# ---------------------------------------------------------------------------

def list_doc_types():
    return {"doc_types": DOC_TYPES}, 200


def health():
    return {
        "status": "healthy",
        "service": "Australian ID Extractor",
        "ocr_device": OCR_DEVICE,
    }, 200


def extract(doc_type, image_bytes, filename="upload.jpg"):
    """Run the full pipeline: validate -> resolution check -> OCR -> fields."""
    if doc_type not in FIELD_SCHEMAS:
        return _error(f"Unknown document type: {doc_type}"), 400

    if not image_bytes:
        return _error("No image data received"), 400

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        return _error(f"File too large. Maximum size is {limit_mb:.0f}MB"), 413

    if not allowed_file(filename):
        return _error(
            "Invalid file type. Allowed types: png, jpg, jpeg, bmp, tiff"
        ), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp_path = temp.name
            temp.write(image_bytes)

        if cv2.imread(temp_path) is None:
            return _error("Uploaded file is not a valid image"), 400

        # Reject outright only if far too small to OCR; otherwise proceed
        # and flag it when below the recommended size.
        resolution = check_image_resolution(temp_path, doc_type.upper())
        if resolution and resolution["below_hard_min"]:
            return {
                "status": "error",
                "icon": "📏",
                "message": (
                    f"Image resolution ({resolution['width']}x{resolution['height']}) "
                    f"is too low for reliable extraction. Minimum: "
                    f"{resolution['hard_min']}px on the long side. "
                    f"Please upload a sharper, higher-resolution photo."
                ),
                "resolution": resolution,
            }, 400

        extracted = get_processors()[doc_type].process_image(temp_path)

        if extracted.get("error") == "BLURRY_IMAGE":
            return {
                "status": "error",
                "icon": "📸",
                "message": _blurry_message(extracted.get("blur_score")),
                "blur_score": extracted.get("blur_score"),
            }, 400

        if "error" in extracted:
            return _error(f"Failed to extract information: {extracted['error']}"), 500

        if resolution and resolution["below_recommended"]:
            status, icon = "warning", "⚠️"
            message = (
                f"Extracted, but image resolution ({resolution['width']}x{resolution['height']}) "
                f"is below the recommended {resolution['recommended']}px (long side) for "
                f"reliable extraction. Some fields may be inaccurate."
            )
        else:
            status, icon = "success", "✅"
            message = "Information extracted successfully!"

        return {
            "status": status,
            "icon": icon,
            "message": message,
            "fields": _build_fields(doc_type, extracted),
            "raw_text": extracted.get("_raw_text", []),
            "raw_boxes": extracted.get("_raw_boxes", []),
            "mrz_detected": extracted.get("_mrz_detected"),
            # "plain" = image passed to OCR essentially untouched;
            # "enhanced" = the photo pipeline was needed as a retry.
            "ocr_mode": extracted.get("_ocr_mode"),
        }, 200

    except Exception as e:  # pragma: no cover - defensive
        import traceback
        print(f"Error in extraction: {traceback.format_exc()}")
        return _error(f"Failed to process image: {e}"), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except PermissionError:
                time.sleep(0.1)
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Warning: could not delete temp file {temp_path}: {e}")


def validate_submission(doc_type, fields):
    """Field-level validation. Returns {field: message} (empty when valid)."""
    errors = {}

    for required in REQUIRED_FIELDS.get(doc_type, []):
        if not (fields.get(required) or "").strip():
            errors[required] = "This field is required."

    for field in FULL_DATE_FIELDS.get(doc_type, []):
        value = (fields.get(field) or "").strip()
        if value and not _FULL_DATE_RE.match(value):
            errors[field] = "Expected a date like DD/MM/YYYY or 01 JAN 2000."

    for field in MONTH_YEAR_FIELDS.get(doc_type, []):
        value = (fields.get(field) or "").strip()
        if value and not _MONTH_YEAR_RE.match(value):
            errors[field] = "Expected a date like MM/YY."

    if doc_type == "medicare":
        value = (fields.get("Medicare Number") or "").strip()
        if value and not _MEDICARE_NO_RE.match(value):
            errors["Medicare Number"] = "Expected 10 digits, e.g. 1234 56789 0."

    return errors


def _persist_record(record):
    """Append-only JSONL stand-in for a real datastore. Replace this one
    function to write somewhere durable."""
    with open(SUBMISSIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def submit(doc_type, fields):
    """Validate the reviewed/edited fields and persist only if they pass."""
    if doc_type not in REQUIRED_FIELDS:
        return _error(f"Unknown document type: {doc_type}"), 400

    fields = fields or {}
    errors = validate_submission(doc_type, fields)
    if errors:
        return {
            "status": "invalid",
            "icon": "⚠️",
            "message": "Please fix the highlighted field(s) above and submit again.",
            "errors": errors,
        }, 400

    submission_id = str(uuid.uuid4())
    record = {
        "id": submission_id,
        "doc_type": doc_type,
        "fields": fields,
        "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    try:
        _persist_record(record)
    except Exception as e:
        return _error(f"Failed to save submission: {e}"), 500

    return {
        "status": "ok",
        "icon": "✅",
        "message": f"Submitted successfully. Reference ID: {submission_id}",
        "id": submission_id,
    }, 200
