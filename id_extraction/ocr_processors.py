"""
OCR processors for Australian ID documents: Medicare card, Passport, and
Driving Licence, built on PaddleOCR 3.x.

Two design decisions matter most here:

1. MINIMAL IMAGE PREPROCESSING BY DEFAULT. Heavy "enhancement" (illumination
   normalization + CLAHE + blur) helps photos of cards taken with a phone,
   but measurably HURTS clean digital card images: it softens character
   edges, which makes the recognizer merge characters and drop the spaces
   between words. So the default path only rescales the image into an
   OCR-friendly size range and leaves the pixels alone -- matching the
   behaviour you get calling ocr.predict() directly on the file. The photo
   pipeline is kept as an automatic RETRY for when the plain pass finds
   almost no text (see OCR_ENHANCE_MODE).

2. SPATIAL (BOUNDING-BOX) FIELD EXTRACTION. ID cards are 2-D layouts, not
   reading-order documents: on an NSW licence, "Licence Class" and
   "Conditions" sit side by side with their values on the line BELOW each
   of them. Matching a label and then taking the next text in reading
   order gets this wrong every time (it reads Conditions as the licence
   class value). Instead every label is located by its bounding box, and
   its value is the nearest text box to its right on the same line, or
   directly below it in the same column -- with other labels excluded as
   candidates and treated as blockers.

Device selection is automatic: CUDA if paddlepaddle was built with CUDA
support and a GPU is visible, otherwise CPU (e.g. on a Mac M1). Override
with PADDLEOCR_DEVICE ("cpu", "gpu:1", ...) and PADDLE_OCR_GPU_INDEX.
"""

import re
import os
import tempfile
import datetime
import cv2
from paddleocr import PaddleOCR

# ---------------------------------------------------------------------------
# Device / OCR engine configuration
# ---------------------------------------------------------------------------


def _detect_device():
    """Auto-detect CUDA availability; PADDLEOCR_DEVICE always overrides.

    Note this checks whether *paddlepaddle* was built with CUDA support and
    can see a GPU -- a plain `pip install paddlepaddle` (CPU-only wheel)
    correctly reports no CUDA here even on a machine with an NVIDIA GPU.
    On macOS this always resolves to CPU: there is no CUDA on a Mac.
    """
    env_device = os.environ.get("PADDLEOCR_DEVICE")
    if env_device:
        return env_device

    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            gpu_index = os.environ.get("PADDLE_OCR_GPU_INDEX", "0")
            return f"gpu:{gpu_index}"
    except Exception as e:
        print(f"[OCR] CUDA detection failed ({e}), falling back to CPU", flush=True)

    return "cpu"


OCR_DEVICE = _detect_device()
print(f"[OCR] Using device: {OCR_DEVICE}", flush=True)


def build_ocr_kwargs():
    """Common PaddleOCR constructor kwargs, adjusted for the selected device."""
    kwargs = dict(
        lang="en",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        device=OCR_DEVICE,
    )
    if OCR_DEVICE == "cpu":
        # Known PaddlePaddle 3.3.x CPU regression: the default
        # enable_mkldnn=True raises "NotImplementedError:
        # ConvertPirAttribute2RuntimeAttribute not support
        # [pir::ArrayAttribute<pir::DoubleAttribute>]" during text-detection
        # inference on CPU. Disabling oneDNN avoids it.
        # See: https://github.com/PaddlePaddle/PaddleOCR/issues/17955
        kwargs["enable_mkldnn"] = False
    return kwargs


# "auto"   -> plain pass, retry with photo enhancement only if barely any
#             text was found (default, best for clean card images)
# "never"  -> never enhance
# "always" -> always use the photo enhancement pipeline (phone photos)
_ENHANCE_MODE = os.environ.get("OCR_ENHANCE_MODE", "auto").lower()

# Below this many detected text boxes, the plain pass is treated as a
# failure worth retrying with enhancement.
_MIN_ITEMS_BEFORE_RETRY = 4


# ---------------------------------------------------------------------------
# Pre-flight image checks (run BEFORE OCR)
# ---------------------------------------------------------------------------

# Thresholds are deliberately permissive. Images are upscaled to ~960px on
# the long side before OCR, and PaddleOCR reads small, clean card scans
# perfectly well after that -- so these only exist to catch input that is
# genuinely unusable (thumbnails), not to gate ordinary web-sized samples.
# "recommended" is advisory only: extraction still runs, the caller just
# gets a warning alongside the result.
_RESOLUTION_THRESHOLDS = {
    "MEDICARE": {"hard_min": 180, "recommended": 400},
    "PASSPORT": {"hard_min": 180, "recommended": 500},
    "LICENCE": {"hard_min": 180, "recommended": 500},
}


def check_image_resolution(image_path, doc_type):
    """Check an image's pixel dimensions against guidance thresholds for
    the given document type ("MEDICARE", "PASSPORT" or "LICENCE").

    Returns None if the image can't be read. Otherwise a dict:
        {"width", "height", "long_side", "hard_min", "recommended",
         "below_hard_min": bool, "below_recommended": bool}
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    long_side = max(h, w)
    thresholds = _RESOLUTION_THRESHOLDS.get(doc_type, _RESOLUTION_THRESHOLDS["MEDICARE"])
    return {
        "width": w,
        "height": h,
        "long_side": long_side,
        "hard_min": thresholds["hard_min"],
        "recommended": thresholds["recommended"],
        "below_hard_min": long_side < thresholds["hard_min"],
        "below_recommended": long_side < thresholds["recommended"],
    }


def _is_image_too_blurry(image_path, threshold=50):
    """Laplacian-variance blur check. Clean card scans score in the
    thousands, so the threshold only catches genuinely smeared photos."""
    img = cv2.imread(image_path)
    if img is None:
        return True, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, score


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

_MIN_SIDE = 960
_MAX_SIDE = 2000


def _resize_for_ocr(image_path):
    """Default path: scale into an OCR-friendly size range and change
    NOTHING else. Returns the original path untouched when the image is
    already in range, so no pixels are resampled at all.

    Writes PNG rather than JPEG so an upscaled image doesn't pick up a
    fresh round of compression artifacts on top of the enlarged ones.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        h, w = img.shape[:2]
        max_side = max(h, w)
        if _MIN_SIDE <= max_side <= _MAX_SIDE:
            return image_path

        if max_side < _MIN_SIDE:
            scale = _MIN_SIDE / max_side
            interp = cv2.INTER_CUBIC
        else:
            scale = _MAX_SIDE / max_side
            interp = cv2.INTER_AREA

        resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.close()
        cv2.imwrite(temp_file.name, resized)
        return temp_file.name

    except Exception as e:
        print(f"[OCR] resize failed ({e}), using original image", flush=True)
        return image_path


def _enhance_for_ocr(image_path):
    """Photo-oriented enhancement: grayscale + illumination normalization +
    mild CLAHE. Helps a phone photo with uneven lighting; hurts a clean
    digital scan (it softens glyph edges), which is why it is a fallback
    rather than the default. Note there is deliberately NO Gaussian blur
    here -- that was measurably destroying character edges.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side < _MIN_SIDE:
            scale = _MIN_SIDE / max_side
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        elif max_side > _MAX_SIDE:
            scale = _MAX_SIDE / max_side
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.medianBlur(gray, 31)
        bg[bg == 0] = 1
        norm = cv2.divide(gray, bg, scale=255)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(norm)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.close()
        cv2.imwrite(temp_file.name, enhanced)
        return temp_file.name

    except Exception as e:
        print(f"[OCR] enhancement failed ({e}), using original image", flush=True)
        return image_path


# ---------------------------------------------------------------------------
# OCR result -> spatial items
# ---------------------------------------------------------------------------

def _get_items(result):
    """Extract [{'text', 'x1','y1','x2','y2', 'xc','yc'}, ...] from a
    PaddleOCR predict() result, preserving bounding boxes. Supports
    'rec_boxes' (axis-aligned) and 'rec_polys'/'dt_polys' (4-point quads).

    If no spatial data is available at all, synthetic boxes are generated
    in reading order so downstream logic still functions (degraded to
    sequential behaviour) rather than returning nothing.
    """
    items = []
    if not result:
        return items

    try:
        res = result[0] if isinstance(result, list) and len(result) > 0 else result
        if not isinstance(res, dict):
            return items

        texts = res.get("rec_texts") or []
        boxes = res.get("rec_boxes")
        polys = res.get("rec_polys") or res.get("dt_polys")

        for i, t in enumerate(texts):
            box = None
            if boxes is not None and i < len(boxes):
                b = boxes[i]
                box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            elif polys is not None and i < len(polys):
                pts = polys[i]
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                box = (min(xs), min(ys), max(xs), max(ys))

            if box is None:
                # No geometry from the model -- fall back to a synthetic
                # single-column layout in reading order.
                box = (0.0, i * 20.0, 100.0, i * 20.0 + 16.0)

            items.append({
                "text": str(t),
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "xc": (box[0] + box[2]) / 2.0,
                "yc": (box[1] + box[3]) / 2.0,
            })

    except Exception as e:
        print(f"[OCR] error reading result items: {e}", flush=True)

    return items


def _norm(text):
    """Uppercase + normalize separators/whitespace for label matching."""
    t = str(text).upper().replace("：", ":").replace("|", " ")
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Fuzzy label matching
#
# OCR garbles label text on small/low-contrast cards -- observed in real
# runs: "VALID TO" -> "VALDTO", "Address:" -> "eset". Matching labels by
# exact spelling silently drops the whole field, so labels are matched by
# edit distance against a canonical keyword instead, with the tolerance
# scaled to keyword length (short keywords must match exactly, or "SEX"
# would match "SET"/"SIX").
# ---------------------------------------------------------------------------

def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def _label_tolerance(keyword):
    n = len(keyword)
    if n <= 4:
        return 0
    if n <= 8:
        return 1
    return 2


def _match_label(norm, keywords):
    """Fuzzy-match any keyword against the START of `norm`, ignoring
    non-letters (so "Conditions:S" matches the keyword "CONDITIONS").

    Returns (distance, end_index_in_norm) for the best match, or None.
    end_index_in_norm is where the label stops in the ORIGINAL string, so
    the caller can read an inline value that follows it.
    """
    letters = []
    positions = []
    for i, ch in enumerate(norm):
        if ch.isalpha():
            letters.append(ch)
            positions.append(i)
    if not letters:
        return None
    letters = "".join(letters)

    best = None
    for keyword in keywords:
        tol = _label_tolerance(keyword)
        low = max(1, len(keyword) - tol)
        high = min(len(letters), len(keyword) + tol)
        for length in range(low, high + 1):
            distance = _levenshtein(letters[:length], keyword)
            if distance <= tol:
                # prefer the closest match, then the longest label consumed
                candidate = (distance, -length, positions[length - 1] + 1)
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        return None
    return best[0], best[2]


def _median_line_height(items):
    heights = [it["y2"] - it["y1"] for it in items if it["y2"] > it["y1"]]
    if not heights:
        return 20.0
    heights.sort()
    return heights[len(heights) // 2]


def _cluster_rows(items, row_band):
    """Group items into visual rows by y-proximity, each row sorted
    left-to-right. Rows are returned top-to-bottom."""
    if not items:
        return []
    items_sorted = sorted(items, key=lambda it: it["yc"])
    rows = []
    current = [items_sorted[0]]
    current_y = items_sorted[0]["yc"]
    for it in items_sorted[1:]:
        if abs(it["yc"] - current_y) <= row_band:
            current.append(it)
            current_y = sum(x["yc"] for x in current) / len(current)
        else:
            rows.append(sorted(current, key=lambda x: x["x1"]))
            current = [it]
            current_y = it["yc"]
    rows.append(sorted(current, key=lambda x: x["x1"]))
    return rows


# ---------------------------------------------------------------------------
# Spatial label -> value reader (the core of the 2-D layout handling)
# ---------------------------------------------------------------------------

class SpatialLabelReader:
    """Finds each field's label box, then the value box that belongs to it.

    A value is either:
      - inline    ("Conditions: S" in a single detected box), or
      - to the RIGHT of the label on the same visual line, or
      - directly BELOW the label in the same column.

    Other labels are never returned as values, and a label sitting between
    a label and a candidate blocks that candidate -- which is what stops
    "Licence Class" from stealing the value that belongs to "Licence No."
    on a card where those labels are stacked.
    """

    def __init__(self, items, label_keywords):
        self.items = items
        self.keywords = label_keywords
        self.norms = [_norm(it["text"]) for it in items]
        self.line_height = _median_line_height(items) if items else 20.0

        # field -> index of the first item recognised as its label
        self.label_index = {}
        # every index that is a label of ANY field (never usable as a value)
        self.label_indices = set()
        # index -> where the label text ends, for reading an inline value
        self.label_ends = {}

        # Each box is assigned to its BEST-matching field rather than the
        # first field that happens to match. Without this, fuzzy matching
        # lets "LICENCE CLASS" match the "LICENCENO" keyword within
        # tolerance and steal the licence-number slot.
        for i, norm in enumerate(self.norms):
            best = None  # (distance, field, end)
            for field, keywords in label_keywords.items():
                match = _match_label(norm, keywords)
                if match is None:
                    continue
                distance, end = match
                if best is None or distance < best[0]:
                    best = (distance, field, end)
            if best is None:
                continue
            _, field, end = best
            self.label_indices.add(i)
            self.label_ends[i] = end
            if field not in self.label_index:
                self.label_index[field] = i

    # -- geometry helpers ---------------------------------------------------

    def _column_overlap(self, a, b):
        return min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])

    def _blocked_below(self, label, cand):
        """Another label sits between `label` and `cand` in the column."""
        for k in self.label_indices:
            other = self.items[k]
            if other is label or other is cand:
                continue
            if label["y2"] - 1 <= other["yc"] <= cand["y1"] + 1:
                if self._column_overlap(other, cand) > 0 or self._column_overlap(other, label) > 0:
                    return True
        return False

    def _blocked_right(self, label, cand):
        """Another label sits between `label` and `cand` on the same line."""
        row_tol = max(self.line_height * 0.6, 6.0)
        for k in self.label_indices:
            other = self.items[k]
            if other is label or other is cand:
                continue
            if abs(other["yc"] - label["yc"]) <= row_tol:
                if label["x2"] - 1 <= other["x1"] and other["x2"] <= cand["x1"] + 1:
                    return True
        return False

    def _nearest_value_index(self, label_idx, exclude=()):
        label = self.items[label_idx]
        lh = self.line_height
        row_tol = max(lh * 0.6, 6.0)
        best = None  # (priority, distance, index) -- priority 0 = right, 1 = below

        for j, it in enumerate(self.items):
            if j == label_idx or j in self.label_indices or j in exclude:
                continue

            # Same line, to the right of the label
            if abs(it["yc"] - label["yc"]) <= row_tol and it["x1"] >= label["x2"] - 2:
                dist = it["x1"] - label["x2"]
                if dist <= lh * 10 and not self._blocked_right(label, it):
                    cand = (0, dist, j)
                    if best is None or cand < best:
                        best = cand
                continue

            # Below the label, in the same column
            dy = it["y1"] - label["y2"]
            if -row_tol * 0.5 <= dy <= lh * 2.2:
                aligned = (
                    self._column_overlap(label, it) > 0
                    or abs(it["x1"] - label["x1"]) <= lh * 1.5
                )
                if aligned and not self._blocked_below(label, it):
                    cand = (1, max(dy, 0.0), j)
                    if best is None or cand < best:
                        best = cand

        return best[2] if best else None

    # -- public API ---------------------------------------------------------

    def inline_value(self, field):
        """Value packed into the label's own box, e.g. 'Conditions: S'.

        A bilingual tail is NOT a value: a passport prints
        "Surname/Nom" or "Passport No./No du passeport" as one box with
        the actual value in a separate box below, so the translated half
        is stripped and the caller falls through to the spatial lookup.
        """
        idx = self.label_index.get(field)
        if idx is None:
            return None
        end = self.label_ends.get(idx)
        if end is None:
            return None
        rest = self.norms[idx][end:]
        rest = re.sub(r"^[\s:\.\-]*/\s*[A-Z][A-Z '\.]*", "", rest)
        rest = rest.strip(" :-\t.")
        return rest or None

    def value_for(self, field, multiline=False):
        """Inline value if present, otherwise the nearest right/below box.

        multiline=True also pulls in the lines stacked directly under the
        first value box (for wrapped values like an address or a
        multi-line 'Valid in' list).
        """
        idx = self.label_index.get(field)
        if idx is None:
            return None

        inline = self.inline_value(field)
        if inline:
            return inline

        j = self._nearest_value_index(idx)

        if j is None:
            # Last resort: the very next detected box. Detection order is
            # roughly reading order, so a label's value usually follows it
            # immediately. Deliberately only idx+1 and never across another
            # label -- scanning further ahead would let "Licence No." on a
            # card where its value is missing grab an unrelated field's
            # value several boxes later.
            nxt = idx + 1
            if nxt < len(self.items) and nxt not in self.label_indices:
                j = nxt

        if j is None:
            return None

        parts = [self.items[j]["text"].strip()]
        if multiline:
            current = j
            used = {j}
            while True:
                nxt = self._continuation_index(current, used)
                if nxt is None:
                    break
                parts.append(self.items[nxt]["text"].strip())
                used.add(nxt)
                current = nxt

        return " ".join(p for p in parts if p) or None

    def _continuation_index(self, idx, used):
        """The next wrapped line directly under items[idx], same column."""
        cur = self.items[idx]
        lh = self.line_height
        best = None
        for j, it in enumerate(self.items):
            if j in used or j in self.label_indices:
                continue
            dy = it["y1"] - cur["y2"]
            if 0 <= dy <= lh * 0.9 and abs(it["x1"] - cur["x1"]) <= lh * 1.2:
                if best is None or dy < best[0]:
                    best = (dy, j)
        return best[1] if best else None


# ---------------------------------------------------------------------------
# Base processor
# ---------------------------------------------------------------------------

class BaseIDProcessor:
    DOC_TYPE = None  # "MEDICARE" | "PASSPORT" | "LICENCE"

    def __init__(self, ocr_engine=None):
        """ocr_engine: an already-constructed PaddleOCR instance can be
        shared across processors (avoids loading the model 3x)."""
        self.ocr = ocr_engine or PaddleOCR(**build_ocr_kwargs())

    def _run_ocr(self, image_path, temp_files):
        """Plain pass first; retry with photo enhancement only if the plain
        pass found almost nothing. Returns (items, mode_used)."""
        items = []

        if _ENHANCE_MODE != "always":
            prepared = _resize_for_ocr(image_path)
            if prepared != image_path:
                temp_files.append(prepared)
            items = _get_items(self.ocr.predict(input=prepared))
            if _ENHANCE_MODE == "never" or len(items) >= _MIN_ITEMS_BEFORE_RETRY:
                return items, "plain"
            print(
                f"[OCR] plain pass found only {len(items)} text box(es); "
                f"retrying with photo enhancement",
                flush=True,
            )

        enhanced = _enhance_for_ocr(image_path)
        if enhanced != image_path:
            temp_files.append(enhanced)
        enhanced_items = _get_items(self.ocr.predict(input=enhanced))

        if len(enhanced_items) > len(items):
            return enhanced_items, "enhanced"
        return items, "plain"

    def process_image(self, image_path):
        temp_files = []
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}

            too_blurry, score = _is_image_too_blurry(image_path)
            if too_blurry:
                return {"error": "BLURRY_IMAGE", "blur_score": round(score, 2)}

            items, mode = self._run_ocr(image_path, temp_files)
            if not items:
                return {"error": "No text detected in image"}

            result = self._process_ocr_items(items)
            result["_raw_text"] = [it["text"] for it in items]
            # Box geometry, exposed for debugging: when a field extracts
            # wrongly, the text alone can't show WHY (which boxes the
            # recognizer merged or split, and where they sit relative to
            # each other) -- the coordinates can.
            result["_raw_boxes"] = [
                {
                    "text": it["text"],
                    "x": round(it["x1"]), "y": round(it["y1"]),
                    "w": round(it["x2"] - it["x1"]), "h": round(it["y2"] - it["y1"]),
                }
                for it in items
            ]
            result["_ocr_mode"] = mode
            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            for path in temp_files:
                try:
                    if path != image_path and os.path.exists(path):
                        os.unlink(path)
                except Exception as e:
                    print(f"Warning: could not delete temp file {path}: {e}")

    def _process_ocr_items(self, items):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared value cleanup
# ---------------------------------------------------------------------------

_DATE_TOKEN_RE = re.compile(
    r"\d{1,2}\s*[A-Z]{3,9}\s*\d{2,4}"          # 01 JAN 2000
    r"|\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}"  # 24/3/1937
)
# The lookarounds stop this matching a FRAGMENT of a full date: without
# them, "Birthdate:24/3/1937" yields "3/1937" and gets mistaken for a
# Medicare expiry.
_MONTH_YEAR_RE = re.compile(r"(?<![\d/])\d{1,2}\s*/\s*\d{2,4}(?![\d/])")
_SHORT_CODE_RE = re.compile(r"^[A-Z0-9]{1,4}\b")


def _clean_date(value):
    if not value:
        return None
    m = _DATE_TOKEN_RE.search(value.upper())
    return m.group(0).strip() if m else value


def _clean_short_code(value):
    if not value:
        return None
    m = _SHORT_CODE_RE.match(value.upper().strip())
    return m.group(0) if m else value


# ---------------------------------------------------------------------------
# Medicare card processor
# ---------------------------------------------------------------------------

class MedicareProcessor(BaseIDProcessor):
    DOC_TYPE = "MEDICARE"

    def _process_ocr_items(self, items):
        line_height = _median_line_height(items)
        rows = _cluster_rows(items, max(line_height * 0.7, 8.0))

        data = {
            "document_type": "Medicare",
            "Medicare Number": None,
            "Valid To": None,
        }
        for i in range(1, 6):
            data[f"Cardholder {i}"] = None

        cardholders = {}

        for row in rows:
            row_text = " ".join(it["text"] for it in row).strip()
            norm = _norm(row_text)

            # --- Medicare number: 10 digits on one line, however the
            # recognizer chose to split them into boxes ("1234 56789 0",
            # or three separate boxes). Reject rows with letters so a
            # name row can't be mistaken for it.
            digits = re.sub(r"\D", "", row_text)
            if data["Medicare Number"] is None and len(digits) == 10:
                if not re.search(r"[A-Z]", norm.replace("MEDICARE", "")):
                    data["Medicare Number"] = f"{digits[0:4]} {digits[4:9]} {digits[9]}"
                    continue

            # --- "VALID TO 11/10". Matched fuzzily: a real run came back
            # as "VALDTO", and an exact "VALID" test silently dropped the
            # field entirely.
            if _match_label(norm, ["VALIDTO", "VALID"]) is not None:
                m = _MONTH_YEAR_RE.search(norm)
                if m:
                    data["Valid To"] = m.group(0).replace(" ", "")
                continue

            # --- Cardholder rows: a leading reference digit 1-9 followed
            # by the name. Works whether the digit is its own box or part
            # of the same box as the name.
            m = re.match(r"^([1-9])[\.\s]+([A-Z][A-Z'\-\s]{1,40})$", norm)
            if m:
                cardholders[int(m.group(1))] = re.sub(r"\s+", " ", m.group(2)).strip()

        for ref, name in cardholders.items():
            if 1 <= ref <= 5:
                data[f"Cardholder {ref}"] = name

        # Fallback: number split across rows / not matched above
        if data["Medicare Number"] is None:
            joined = " ".join(it["text"] for it in items)
            m = re.search(r"\b(\d{4})\s?(\d{5})\s?(\d)\b", joined)
            if m:
                data["Medicare Number"] = f"{m.group(1)} {m.group(2)} {m.group(3)}"

        # Final fallback: the expiry is the only MM/YY date printed on a
        # Medicare card, so if the label was too garbled to recognise at
        # all, the lone month/year token is still unambiguous.
        if data["Valid To"] is None:
            for it in items:
                norm = _norm(it["text"])
                # Skip anything holding a full date -- a MM/YY match inside
                # "24/3/1937" is a fragment, not an expiry.
                if _DATE_TOKEN_RE.search(norm):
                    continue
                if re.search(r"\d{5,}", norm.replace(" ", "")):
                    continue  # a long digit run is the card number
                m = _MONTH_YEAR_RE.search(norm)
                if m:
                    data["Valid To"] = m.group(0).replace(" ", "")
                    break

        return data


# ---------------------------------------------------------------------------
# Passport processor (MRZ-first, spatial label fallback)
# ---------------------------------------------------------------------------

class PassportProcessor(BaseIDProcessor):
    DOC_TYPE = "PASSPORT"

    _MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{20,}$")

    # Bio-page labels. On a real passport the English label and its
    # bilingual counterpart share one detected box ("Passport No./No du
    # passeport") and the value is a SEPARATE box below -- which the
    # spatial reader handles without needing to parse the translation.
    _LABEL_KEYWORDS = {
        "Surname": ["SURNAME"],
        "Given Names": ["GIVENNAMES", "GIVENNAME"],
        "Passport Number": ["PASSPORTNO", "PASSPORTNUMBER"],
        "Nationality": ["NATIONALITY"],
        "Date of Birth": ["DATEOFBIRTH", "BIRTHDATE"],
        "Sex": ["SEX"],
        "Place of Birth": ["PLACEOFBIRTH"],
        "Date of Issue": ["DATEOFISSUE"],
        "Date of Expiry": ["DATEOFEXPIRY", "DATEOFEXPIRATION"],
    }

    _SEX_TOKEN_RE = re.compile(r"\b[MF]\b")
    _ID_TOKEN_RE = re.compile(r"[A-Z0-9]{5,12}")

    def _find_mrz_lines(self, texts):
        """MRZ lines are long runs of A-Z0-9< with several '<' fillers --
        distinctive enough to find regardless of detection order."""
        candidates = []
        for line in texts:
            norm = line.upper().replace(" ", "")
            if self._MRZ_LINE_RE.match(norm) and norm.count("<") >= 3:
                candidates.append(norm)
        return candidates[:2]  # TD3 passports have exactly two MRZ lines

    def _parse_mrz(self, mrz_lines):
        fields = {}
        if not mrz_lines:
            return fields

        line1 = next((l for l in mrz_lines if l.startswith("P")), None)
        if line1:
            body = line1[5:] if line1.startswith("P<") else line1[2:]
            parts = re.split(r"<<+", body)
            if parts:
                fields["Surname"] = parts[0].replace("<", " ").strip()
            if len(parts) > 1:
                fields["Given Names"] = parts[1].replace("<", " ").strip()

        line2 = next((l for l in mrz_lines if l != line1), None)
        if line2:
            m = re.match(r"^([A-Z0-9<]{9})(\d)([A-Z]{3})(\d{6})(\d)([MF<])(\d{6})(\d)", line2)
            if m:
                fields["Passport Number"] = m.group(1).replace("<", "").strip()
                fields["Nationality"] = m.group(3)
                fields["Date of Birth"] = m.group(4)
                fields["Sex"] = m.group(6)
                fields["Date of Expiry"] = m.group(7)

        return fields

    @staticmethod
    def _format_yymmdd(yymmdd, assume_century=None):
        """MRZ dates are YYMMDD with no century digit.

        assume_century=None uses the nearest-century heuristic (right for a
        date of birth, which could be either century). Expiry dates must
        NOT use it -- a passport expiring in "30" means 2030, not 1930 --
        so callers pass assume_century="20" for those.
        """
        if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
            return yymmdd
        yy, mm, dd = yymmdd[0:2], yymmdd[2:4], yymmdd[4:6]
        if assume_century:
            century = assume_century
        else:
            current_yy = datetime.datetime.now().year % 100
            century = "19" if int(yy) > current_yy else "20"
        return f"{dd}/{mm}/{century}{yy}"

    def _process_ocr_items(self, items):
        texts = [it["text"] for it in items]
        data = {
            "document_type": "Passport",
            "Surname": None,
            "Given Names": None,
            "Passport Number": None,
            "Nationality": None,
            "Date of Birth": None,
            "Sex": None,
            "Place of Birth": None,
            "Date of Issue": None,
            "Date of Expiry": None,
        }

        mrz_fields = self._parse_mrz(self._find_mrz_lines(texts))
        if mrz_fields:
            data["Surname"] = mrz_fields.get("Surname")
            data["Given Names"] = mrz_fields.get("Given Names")
            data["Passport Number"] = mrz_fields.get("Passport Number")
            data["Nationality"] = mrz_fields.get("Nationality")
            if mrz_fields.get("Date of Birth"):
                data["Date of Birth"] = self._format_yymmdd(mrz_fields["Date of Birth"])
            data["Sex"] = mrz_fields.get("Sex")
            if mrz_fields.get("Date of Expiry"):
                data["Date of Expiry"] = self._format_yymmdd(
                    mrz_fields["Date of Expiry"], assume_century="20"
                )

        # Spatial label fallback for whatever the MRZ didn't provide (or
        # for a photo where the MRZ is cropped off / unreadable)
        reader = SpatialLabelReader(items, self._LABEL_KEYWORDS)
        for field in self._LABEL_KEYWORDS:
            if data.get(field):
                continue
            value = reader.value_for(field)
            if value:
                data[field] = re.sub(r"\s+", " ", value).strip(" :")

        for date_field in ("Date of Birth", "Date of Issue", "Date of Expiry"):
            data[date_field] = _clean_date(data.get(date_field))

        if data.get("Sex"):
            m = self._SEX_TOKEN_RE.search(data["Sex"].upper())
            data["Sex"] = m.group(0) if m else data["Sex"]

        if data.get("Passport Number"):
            m = self._ID_TOKEN_RE.search(re.sub(r"\s+", "", data["Passport Number"].upper()))
            data["Passport Number"] = m.group(0) if m else data["Passport Number"]

        data["_mrz_detected"] = bool(mrz_fields)
        return data


# ---------------------------------------------------------------------------
# Driving licence processor (spatial; layouts vary a lot by state)
# ---------------------------------------------------------------------------

class DrivingLicenceProcessor(BaseIDProcessor):
    DOC_TYPE = "LICENCE"

    # Canonical label spellings (letters only). Matched fuzzily, so OCR
    # noise in the label itself doesn't drop the field.
    _LABEL_KEYWORDS = {
        "Licence Number": ["LICENCENO", "LICENSENO", "LICENCENUMBER", "LICENSENUMBER"],
        "Licence Class": ["LICENCECLASS", "LICENSECLASS", "CLASS"],
        "Conditions": ["CONDITIONS", "CONDITION"],
        "Donor": ["DONOR"],
        "Date of Birth": ["DATEOFBIRTH", "BIRTHDATE", "DOB"],
        "Expiry Date": ["EXPIRYDATE", "EXPIRES", "EXPIRY", "EXPIRATIONDATE"],
        "Card Number": ["CARDNUMBER", "CARDNO"],
        "Valid In": ["VALIDIN"],
        "Address": ["ADDRESS"],
    }

    # Belt-and-braces: if a field is still empty after the spatial pass,
    # look for "label<sep>value" inside any single box. Covers boxes whose
    # geometry confused the spatial reader.
    _TEXT_FALLBACKS = {
        "Licence Class": r"CLASS\s*[:\-]?\s*([A-Z0-9]{1,3})\b",
        "Conditions": r"C[O0]ND[I1L]?T[I1L]?[O0]NS?\s*[:\-]?\s*([A-Z0-9]{1,3})\b",
        "Donor": r"D[O0]N[O0]R\s*[:\-]?\s*([A-Z0-9]{1,3})\b",
        "Licence Number": r"LICEN[CS]E\s*N[O0][\.\s:]*([A-Z0-9][A-Z0-9\s]{4,20})",
    }

    # Words that are card furniture or annotation callouts, never a person's
    # name. (The NSW specimen image is an annotated diagram, so it literally
    # contains the words "Given Name" / "Family Name" / "Middle Name".)
    _NAME_STOPWORDS = {
        "DRIVER", "LICENCE", "LICENSE", "AUSTRALIA", "AUSTRALIAN", "NEW", "SOUTH",
        "WALES", "VICTORIA", "QUEENSLAND", "TASMANIA", "TERRITORY", "CAPITAL",
        "WESTERN", "NORTHERN", "GIVEN", "FAMILY", "MIDDLE", "NAME", "NAMES",
        "CARD", "NUMBER", "CLASS", "CONDITIONS", "DONOR", "DATE", "BIRTH",
        "EXPIRY", "EXPIRES", "ADDRESS", "VALID", "SIGNATURE", "STATE",
    }

    # A SEPARATE, narrower list for the address block. Place names are card
    # furniture when deciding "is this a person's name?" but are perfectly
    # legitimate inside an address ("c/o The Open Road, Australia"), so the
    # name stoplist must not be reused here.
    _ADDRESS_STOPWORDS = {
        "GIVEN", "FAMILY", "MIDDLE", "NAME", "NAMES", "SIGNATURE",
        "CARD", "NUMBER", "CLASS", "CONDITIONS", "DONOR",
        "EXPIRY", "EXPIRES", "LICENCE", "LICENSE", "DRIVER",
    }

    _STREET_HINT_RE = re.compile(
        r"\b(ST|STREET|RD|ROAD|AVE|AVENUE|DR|DRIVE|LANE|LN|CT|COURT|PL|PLACE|"
        r"PLAZA|HWY|HIGHWAY|PDE|PARADE|CRES|CRESCENT|TCE|TERRACE|BLVD|BOX|UNIT)\b"
    )
    # A locality line needs BOTH a state abbreviation and a 4-digit postcode.
    # Matching on a bare 4-digit number would treat any year ("01 JAN 2000")
    # as an address line.
    _STATE_POSTCODE_RE = re.compile(
        r"\b(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\b(?=.*\b\d{4}\b)"
        r"|\b\d{4}\b(?=.*\b(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\b)"
    )

    def _process_ocr_items(self, items):
        reader = SpatialLabelReader(items, self._LABEL_KEYWORDS)

        data = {"document_type": "Driving Licence"}
        for field in self._LABEL_KEYWORDS:
            data[field] = None

        for field in self._LABEL_KEYWORDS:
            multiline = field in ("Address", "Valid In")
            value = reader.value_for(field, multiline=multiline)
            if value:
                data[field] = re.sub(r"\s+", " ", value).strip(" :")

        # Per-box regex fallback for anything the spatial pass missed
        for field, pattern in self._TEXT_FALLBACKS.items():
            if data.get(field):
                continue
            for norm in reader.norms:
                m = re.search(pattern, norm)
                if m:
                    data[field] = m.group(1).strip()
                    break

        for date_field in ("Date of Birth", "Expiry Date"):
            data[date_field] = _clean_date(data.get(date_field))

        for code_field in ("Licence Class", "Conditions", "Donor"):
            data[code_field] = _clean_short_code(data.get(code_field))

        address_indices = set()
        if not data.get("Address"):
            block, address_indices = self._find_address_block(items, reader)
            data["Address"] = block

        data["Name"] = self._find_name(items, reader, address_indices)

        return data

    # -- heuristics for the two unlabelled fields ---------------------------

    def _find_address_block(self, items, reader):
        """Cards often print the address with no 'Address' label at all.
        Find an anchor line that looks like a street/suburb line, then pull
        in the lines stacked immediately above and below it in the same
        column (so 'CENTENNIAL PLAZA', which has no digits of its own,
        still joins its street and suburb lines).
        """
        line_height = reader.line_height
        anchors = []
        for i, it in enumerate(items):
            if i in reader.label_indices:
                continue
            norm = _norm(it["text"])
            if len(norm) < 5:
                continue
            if _DATE_TOKEN_RE.search(norm):
                continue  # an expiry/birth date is not an address line
            has_digit = bool(re.search(r"\d", norm))
            # A street-type word alone is enough: "c/o The Open Road" is a
            # perfectly good address line with no digits in it, and
            # requiring a digit silently dropped the whole address block.
            looks_street = bool(self._STREET_HINT_RE.search(norm))
            looks_locality = bool(self._STATE_POSTCODE_RE.search(norm))
            starts_with_street_number = bool(re.match(r"^\d+[A-Z]?\s+[A-Z]", norm))
            if looks_street or looks_locality or (has_digit and starts_with_street_number):
                anchors.append(i)

        if not anchors:
            return None, set()

        chosen = set(anchors)
        # Grow the block vertically through neighbouring lines in the column
        changed = True
        while changed:
            changed = False
            for i in list(chosen):
                cur = items[i]
                for j, it in enumerate(items):
                    if j in chosen or j in reader.label_indices:
                        continue
                    text = it["text"].strip()
                    norm = _norm(text)
                    if len(norm) < 4 or not re.search(r"[A-Z]{3,}", norm):
                        continue
                    if _DATE_TOKEN_RE.search(norm):
                        continue
                    # Annotation callouts ("Given Name", "Family Name") sit
                    # right above the address on specimen diagrams -- the
                    # stopword list, not letter case, is what keeps them
                    # out (some cards print the address in title case).
                    if any(tok.strip(".,:") in self._ADDRESS_STOPWORDS for tok in norm.split()):
                        continue
                    # A lone short token is OCR noise (a garbled label),
                    # not an address line; a real single-word line like
                    # "AUSTRALIA" is comfortably longer.
                    if len(norm.split()) == 1 and len(norm) < 6:
                        continue
                    gap_below = it["y1"] - cur["y2"]
                    gap_above = cur["y1"] - it["y2"]
                    vertical_ok = (0 <= gap_below <= line_height * 0.9) or (0 <= gap_above <= line_height * 0.9)
                    aligned = abs(it["x1"] - cur["x1"]) <= line_height * 1.2
                    if vertical_ok and aligned:
                        chosen.add(j)
                        changed = True

        ordered = sorted(chosen, key=lambda i: items[i]["y1"])
        lines = [items[i]["text"].strip() for i in ordered]
        return (", ".join(l for l in lines if l) or None), chosen

    def _find_name(self, items, reader, address_indices):
        """A person's name: a title-prefixed line if present, otherwise the
        best 2-4 word line of alphabetic tokens that isn't card furniture,
        an annotation callout, a label, or part of the address block."""
        for it in items:
            text = it["text"].strip()
            if re.match(r"^(MR|MRS|MS|MISS|MX)\.?\s+\S", text, re.IGNORECASE):
                return text

        best = None
        for i, it in enumerate(items):
            if i in reader.label_indices or i in address_indices:
                continue
            text = it["text"].strip()
            if not text or re.search(r"\d", text):
                continue
            tokens = [t for t in re.split(r"\s+", text) if t]
            if not (2 <= len(tokens) <= 4):
                continue
            if not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-\.]*", t) for t in tokens):
                continue
            if any(_norm(t).strip(".") in self._NAME_STOPWORDS for t in tokens):
                continue
            real_words = [t for t in tokens if len(t) >= 3]
            if len(real_words) < 2:
                continue
            score = len(real_words)
            if best is None or score > best[0]:
                best = (score, text)

        return best[1] if best else None
