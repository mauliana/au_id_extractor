# Australian ID Extractor

Extracts structured fields from photos of Australian identity documents —
**Medicare cards**, **passports** and **driving licences** — using
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), then lets a human
review and correct every field before submitting.

Built as a teaching / research reference implementation: the OCR pipeline,
the 2-D layout parsing and the human-in-the-loop review step are each kept
small and readable.

![pipeline](https://img.shields.io/badge/pipeline-check%20%E2%86%92%20OCR%20%E2%86%92%20review%20%E2%86%92%20submit-blue)

---

## How it works

```
upload → pre-flight checks → OCR → spatial field extraction → human review → validate → submit
         (size, resolution,        (PaddleOCR)   (bounding-box       (editable      (per-field
          readability, blur)                      label→value)        form)          rules)
```

Two design decisions carry most of the accuracy:

**1. Minimal image preprocessing.** Heavy "enhancement" (illumination
normalisation + CLAHE + blur) helps photos taken with a phone but measurably
*hurts* clean digital card images — it softens character edges, so the
recogniser merges characters and drops spaces between words. On a real
sample this cost 38% of the edge energy. The default path therefore only
rescales the image into an OCR-friendly size range and leaves the pixels
alone. The photo pipeline is kept as an automatic retry for when the plain
pass finds almost no text (`OCR_ENHANCE_MODE`).

**2. Spatial (bounding-box) field extraction.** ID cards are 2-D layouts,
not reading-order documents. On an NSW licence, `Licence Class` and
`Conditions` sit side by side with each value on the line *below* its own
label — so "take the next text after the label" reads the conditions value
as the licence class every time. Instead each label is located by its
bounding box and its value is the nearest box to its right on the same
line, or directly below it in the same column, with other labels excluded
as candidates *and* treated as blockers.

Labels are matched **fuzzily** (edit distance), because OCR garbles the
label text itself on small images — real observed output includes
`VALID TO` → `VALDTO` and `Address:` → `eset`. Exact matching silently
dropped those whole fields.

---

## Project layout

```
.
├── streamlit_app.py          # UI entry point (Streamlit Cloud runs this)
├── api.py                    # OPTIONAL standalone Flask backend
├── id_extraction/
│   ├── service.py            # all business logic: rules, validation, responses
│   └── ocr_processors.py     # OCR pipeline + per-document field extraction
├── tests/smoke_test.py       # runs without PaddleOCR installed
├── requirements.txt
├── packages.txt              # apt packages for Streamlit Cloud
└── .streamlit/config.toml
```

`streamlit_app.py` is **pure UI**: it contains no document knowledge, no
validation and no message text. It asks the backend what document types
exist, what fields to render and what to display, then renders exactly
that. All rules live in `id_extraction/service.py`.

---

## Running it

### Option A — Streamlit Community Cloud (recommended)

Streamlit Cloud runs a **single process**, so it cannot host a separate
Flask server alongside the UI. The app therefore calls the service
**in-process** by default — no configuration needed.

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app
   pointing at `streamlit_app.py`.
3. Under *Advanced settings*, set Python version to **3.11**.
4. Deploy. The first run downloads the PaddleOCR models, so expect a slow
   cold start (a few minutes); later runs are cached.

> **Memory note.** The free Community Cloud tier gives ~1 GB of RAM, and
> PaddlePaddle plus the OCR models is a heavy import. If the app is killed
> on startup, deploy to a container host with ≥ 2 GB instead (Hugging Face
> Spaces, Render, Fly.io, Cloud Run) — the same repo runs unchanged.

> **Storage note.** Submissions append to `submissions.jsonl`, and the
> Streamlit Cloud filesystem is **ephemeral** — it is wiped on restart or
> redeploy. Replace `service._persist_record()` with a real database for
> anything you need to keep.

### Option B — locally, single process

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Option C — locally, with the Flask backend as a separate process

Useful if you want the API available to other clients too.

```bash
# terminal 1
python api.py                       # http://127.0.0.1:5000

# terminal 2
BACKEND_URL=http://127.0.0.1:5000 streamlit run streamlit_app.py
```

Both options run **the same code** in `service.py`; only the transport
differs.

---

## API reference (Option C)

| Method | Path                 | Purpose                                        |
|--------|----------------------|------------------------------------------------|
| GET    | `/health`            | Liveness + which OCR device is active          |
| GET    | `/doc-types`         | Document types, labels and endpoints for the UI|
| POST   | `/extract/medicare`  | Extract from a Medicare card (`image` file)    |
| POST   | `/extract/passport`  | Extract from a passport (`image` file)         |
| POST   | `/extract/licence`   | Extract from a driving licence (`image` file)  |
| POST   | `/submit`            | Validate + persist `{doc_type, fields}`        |

Every response is display-ready:

```jsonc
{
  "status": "success",           // success | warning | error | ok | invalid
  "icon": "✅",
  "message": "Information extracted successfully!",
  "fields": [                    // ordered, ready to render as a form
    {"key": "Medicare Number", "label": "Medicare Number:", "value": "1234 56789 0"}
  ],
  "raw_text": ["..."],           // every detected text box
  "raw_boxes": [{"text": "...", "x": 25, "y": 105, "w": 165, "h": 20}],
  "ocr_mode": "plain"            // plain | enhanced
}
```

---

## Configuration

| Variable             | Default            | Purpose                                            |
|----------------------|--------------------|----------------------------------------------------|
| `BACKEND_URL`        | *(unset)*          | Point the UI at a separate Flask backend           |
| `PADDLE_OCR_DEVICE`  | auto-detected      | Force `cpu` / `gpu:0`; otherwise CUDA is auto-used  |
| `PADDLE_OCR_GPU_INDEX` | `0`              | Which GPU, when CUDA is available                  |
| `OCR_ENHANCE_MODE`   | `auto`             | `auto` \| `never` \| `always` photo enhancement    |
| `SUBMISSIONS_FILE`   | `submissions.jsonl`| Where accepted submissions are appended            |
| `MAX_UPLOAD_BYTES`   | `5242880` (5 MB)   | Upload size limit                                  |

CUDA is detected automatically: if PaddlePaddle was built with CUDA support
and a GPU is visible it uses `gpu:0`, otherwise it falls back to CPU
(always the case on macOS, which has no CUDA).

---

## Debugging a bad extraction

Expand **"Show raw OCR output"** under the results. It lists every detected
text box with its `x/y/w/h`. That geometry — not the text alone — is what
explains a wrong field: it shows whether the recogniser merged two values
into one box, split one value across two, or placed a label somewhere the
layout logic didn't expect.

`ocr_mode` tells you whether the image went through OCR untouched
(`plain`) or needed the photo-enhancement retry (`enhanced`).

---

## Testing

```bash
python tests/smoke_test.py
```

The tests feed hand-built **bounding-box fixtures** that reconstruct real
card layouts, so they run without PaddleOCR installed and without any real
ID images in the repo. Any change to the extraction logic should be checked
against them.

---

## Limitations

- Driving licence layouts vary considerably between states; the field
  heuristics are tuned against a generic federal-style and an NSW-style
  sample.
- At very low resolution the recogniser itself misreads characters
  (`9/1/13` → `11/1/0`). That is a recognition limit, not a parsing one —
  which is exactly why the review-and-edit step exists before submission.
- Field extraction is best-effort. **Always review before submitting.**

## Privacy

Identity documents are sensitive personal data. Uploads are written to a
temporary file, passed to OCR and deleted immediately afterwards; nothing
is retained except what you explicitly submit. If you deploy this
publicly, add authentication and check your obligations under the
Australian Privacy Act before handling real documents.

## Licence

MIT — see `LICENSE`.
