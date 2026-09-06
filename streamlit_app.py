"""
Streamlit entry point for the Australian ID Extractor.

This file is PURE UI. It contains no document knowledge, no validation and
no message text -- it asks the backend what document types exist, what
fields to render, and what to display, then renders exactly that.

Two backend modes, chosen automatically:

  * IN-PROCESS (default) -- calls id_extraction.service directly. This is
    what Streamlit Community Cloud needs, because it runs a single process
    and cannot host a separate Flask server alongside the UI.
  * HTTP -- set BACKEND_URL (env var or .streamlit/secrets.toml) to point
    at a running api.py, e.g. http://127.0.0.1:5000

Either way the business logic lives in the backend module, never here.
"""

import os
import json

import streamlit as st

st.set_page_config(layout="wide", page_title="Australian ID Extractor", page_icon="🇦🇺")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _backend_url():
    url = os.environ.get("BACKEND_URL")
    if url:
        return url.rstrip("/")
    try:  # secrets.toml is optional; absent on most deployments
        url = st.secrets.get("BACKEND_URL")
        return url.rstrip("/") if url else None
    except Exception:
        return None


BACKEND_URL = _backend_url()


@st.cache_resource(show_spinner="Loading OCR models (first run only)...")
def _load_backend():
    """Import the service once per process. Cached so the PaddleOCR models
    are not reloaded on every Streamlit rerun."""
    from id_extraction import service
    service.get_processors()  # warm the models up front
    return service


def call_doc_types():
    if BACKEND_URL:
        import requests
        r = requests.get(f"{BACKEND_URL}/doc-types", timeout=10)
        return r.json().get("doc_types", [])
    payload, _ = _load_backend().list_doc_types()
    return payload.get("doc_types", [])


def call_extract(doc_type, image_bytes, filename):
    if BACKEND_URL:
        import requests
        r = requests.post(
            f"{BACKEND_URL}/extract/{doc_type}",
            files={"image": (filename, image_bytes, "image/jpeg")},
            timeout=180,
        )
        return r.json()
    payload, _ = _load_backend().extract(doc_type, image_bytes, filename)
    return payload


def call_submit(doc_type, fields):
    if BACKEND_URL:
        import requests
        r = requests.post(
            f"{BACKEND_URL}/submit",
            json={"doc_type": doc_type, "fields": fields},
            timeout=30,
        )
        return r.json()
    payload, _ = _load_backend().submit(doc_type, fields)
    return payload


# Maps the backend's status string to a Streamlit rendering method. The
# API already decided success/warning/error; the UI only picks the widget.
STATUS_TO_WIDGET = {
    "success": "success",
    "warning": "warning",
    "error": "error",
    "ok": "success",
    "invalid": "warning",
}


def show_notification(target, payload):
    """Render whatever the backend said to render."""
    widget = getattr(target, STATUS_TO_WIDGET.get(payload.get("status"), "info"))
    widget(payload.get("message", ""), icon=payload.get("icon"))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("doc_types", None),
    ("doc_type", None),
    ("extracted_fields", None),
    ("extracted_meta", None),
    ("uploaded_image_name", None),
    ("last_notification", None),
    ("submit_errors", None),
    ("submit_result", None),
]:
    st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Header + document type picker
# ---------------------------------------------------------------------------

st.title("🇦🇺 Australian ID Information Extractor")
st.markdown(
    "Upload a **Medicare card**, **Passport**, or **Driving Licence** image "
    "to extract its information, review and edit it, then submit."
)

try:
    if st.session_state.doc_types is None:
        st.session_state.doc_types = call_doc_types()
except Exception as e:
    st.error(f"Could not reach the backend: {e}", icon="🔌")
    if BACKEND_URL:
        st.info(f"Configured backend: `{BACKEND_URL}` — is `python api.py` running?")
    st.stop()

doc_types = st.session_state.doc_types
if not doc_types:
    st.error("Backend returned no document types.", icon="⚠️")
    st.stop()

if st.session_state.doc_type is None:
    st.session_state.doc_type = doc_types[0]["key"]

labels = [d["label"] for d in doc_types]
current_label = next(d["label"] for d in doc_types if d["key"] == st.session_state.doc_type)

chosen_label = st.radio(
    "Document type",
    options=labels,
    index=labels.index(current_label),
    horizontal=True,
)
chosen = next(d for d in doc_types if d["label"] == chosen_label)

if chosen["key"] != st.session_state.doc_type:
    # Switching document type invalidates anything already extracted.
    st.session_state.update(
        doc_type=chosen["key"],
        extracted_fields=None,
        extracted_meta=None,
        uploaded_image_name=None,
        last_notification=None,
        submit_errors=None,
        submit_result=None,
    )

doc_cfg = next(d for d in doc_types if d["key"] == st.session_state.doc_type)
col_left, col_right = st.columns([1, 1])


# ---------------------------------------------------------------------------
# Left: upload + extract
# ---------------------------------------------------------------------------

with col_left:
    st.subheader(f"📷 {doc_cfg['label']} Image")

    uploaded_file = st.file_uploader(
        f"Choose a {doc_cfg['label']} image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image to begin extraction.")
    else:
        if st.session_state.uploaded_image_name != uploaded_file.name:
            st.session_state.update(
                extracted_fields=None,
                extracted_meta=None,
                uploaded_image_name=uploaded_file.name,
                last_notification=None,
                submit_errors=None,
                submit_result=None,
            )

        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption=f"Uploaded {doc_cfg['label']} Image", use_container_width=True)

        extract_clicked = st.button(
            "🔍 Extract Information", type="primary", use_container_width=True
        )
        notification_area = st.empty()

        if extract_clicked:
            st.session_state.submit_errors = None
            st.session_state.submit_result = None
            with st.spinner("Extracting information... Please wait."):
                try:
                    payload = call_extract(
                        st.session_state.doc_type, image_bytes, uploaded_file.name
                    )
                except Exception as e:
                    payload = {
                        "status": "error",
                        "icon": "🛑",
                        "message": f"An unexpected error occurred: {e}",
                    }

            st.session_state.last_notification = payload
            if payload.get("status") in ("success", "warning"):
                st.session_state.extracted_fields = payload.get("fields", [])
                st.session_state.extracted_meta = {
                    "raw_text": payload.get("raw_text"),
                    "raw_boxes": payload.get("raw_boxes"),
                    "mrz_detected": payload.get("mrz_detected"),
                    "ocr_mode": payload.get("ocr_mode"),
                }
            else:
                st.session_state.extracted_fields = None
                st.session_state.extracted_meta = None

        if st.session_state.last_notification:
            show_notification(notification_area, st.session_state.last_notification)


# ---------------------------------------------------------------------------
# Right: editable results + submit
# ---------------------------------------------------------------------------

with col_right:
    st.subheader("📝 Extracted Information")

    if not st.session_state.extracted_fields:
        st.info("📋 Extracted information will appear here after processing an image.")
    else:
        with st.form("edit_form"):
            form_data = {}
            for field in st.session_state.extracted_fields:
                form_data[field["key"]] = st.text_input(
                    field["label"], value=str(field.get("value") or "")
                )

            # Field-level errors from a previous submit, shown right above
            # the buttons so it is clear what still needs fixing.
            if st.session_state.submit_errors:
                for name, message in st.session_state.submit_errors.items():
                    st.error(f"**{name}**: {message}")

            c1, c2, c3 = st.columns(3)
            with c1:
                save_clicked = st.form_submit_button("💾 Save Changes", use_container_width=True)
            with c2:
                json_clicked = st.form_submit_button("📋 Copy JSON", use_container_width=True)
            with c3:
                submit_clicked = st.form_submit_button(
                    "✅ Submit", type="primary", use_container_width=True
                )

            def _persist_edits():
                for field in st.session_state.extracted_fields:
                    field["value"] = form_data.get(field["key"])

            if save_clicked:
                _persist_edits()
                st.success("Changes saved. Keep editing, or submit when ready.", icon="💾")

            if json_clicked:
                st.code(json.dumps(form_data, ensure_ascii=False, indent=2), language="json")
                st.info("Copy the JSON above to use it elsewhere", icon="📋")

            if submit_clicked:
                # Save edits first so a failed submit still shows what was typed.
                _persist_edits()
                try:
                    result = call_submit(st.session_state.doc_type, form_data)
                except Exception as e:
                    result = {
                        "status": "error",
                        "icon": "🛑",
                        "message": f"An unexpected error occurred while submitting: {e}",
                    }
                st.session_state.submit_errors = result.get("errors")
                st.session_state.submit_result = result

        # Outside the form so it survives the rerun.
        if st.session_state.submit_result:
            show_notification(st, st.session_state.submit_result)

        meta = st.session_state.extracted_meta or {}
        if meta.get("raw_text") or meta.get("raw_boxes"):
            with st.expander("🔍 Show raw OCR output (for debugging)"):
                mode = meta.get("ocr_mode")
                if mode:
                    st.caption(
                        "Image sent to OCR essentially untouched (rescale only)."
                        if mode == "plain"
                        else "Plain pass found little text, so photo enhancement was applied as a retry."
                    )
                if meta.get("raw_boxes"):
                    # One row per detected text box with its position. When a
                    # field extracts wrongly, the text alone can't show why --
                    # whether boxes were merged or split, and how they sit
                    # relative to each other, is what explains it.
                    st.dataframe(meta["raw_boxes"], use_container_width=True)
                else:
                    st.text("\n".join(meta.get("raw_text") or []))

        if meta.get("mrz_detected") is not None:
            st.caption(
                "MRZ line detected and used for parsing ✅"
                if meta["mrz_detected"]
                else "No MRZ line detected — fields parsed from labels only ⚠️"
            )


with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Extracts fields from Australian **Medicare cards**, **passports** "
        "and **driving licences** using PaddleOCR."
    )
    st.markdown("**Backend mode**")
    st.code(BACKEND_URL if BACKEND_URL else "in-process (no separate server)", language=None)
    if not BACKEND_URL:
        try:
            st.caption(f"OCR device: `{_load_backend().ocr_device()}`")
        except Exception:
            pass
    st.markdown("---")
    st.caption(
        "Extraction is a best-effort read of a photo — always review the "
        "fields before submitting."
    )
