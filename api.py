"""
Flask HTTP wrapper around id_extraction.service.

Optional: use this when you want the backend to run as a separate process
(self-hosting, or serving other clients besides the Streamlit UI). Point
the UI at it with BACKEND_URL=http://127.0.0.1:5000.

Streamlit Community Cloud runs a single process and cannot host this
alongside the UI -- there, streamlit_app.py calls the service in-process
instead. Both paths execute the same code in service.py.

    python api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from id_extraction import service

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = service.MAX_UPLOAD_BYTES


@app.errorhandler(413)
def request_entity_too_large(error):
    limit_mb = service.MAX_UPLOAD_BYTES / (1024 * 1024)
    return jsonify({
        "status": "error",
        "icon": "❌",
        "message": f"File too large. Maximum size is {limit_mb:.0f}MB",
    }), 413


def _extract(doc_type):
    if "image" not in request.files:
        return jsonify(service._error("No image part in the request")), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify(service._error("No image selected for uploading")), 400

    payload, status = service.extract(doc_type, file.read(), file.filename)
    return jsonify(payload), status


@app.route("/extract/medicare", methods=["POST"])
def extract_medicare():
    """Extract fields from an Australian Medicare card image."""
    return _extract("medicare")


@app.route("/extract/passport", methods=["POST"])
def extract_passport():
    """Extract fields from an Australian passport image."""
    return _extract("passport")


@app.route("/extract/licence", methods=["POST"])
def extract_licence():
    """Extract fields from an Australian driving licence image."""
    return _extract("licence")


@app.route("/doc-types", methods=["GET"])
def doc_types():
    """Document types, labels and field order for the UI to render."""
    payload, status = service.list_doc_types()
    return jsonify(payload), status


@app.route("/submit", methods=["POST"])
def submit():
    """Validate reviewed/edited fields and persist them if they pass."""
    body = request.get_json(silent=True) or {}
    payload, status = service.submit(body.get("doc_type"), body.get("fields"))
    return jsonify(payload), status


@app.route("/health", methods=["GET"])
def health():
    payload, status = service.health()
    return jsonify(payload), status


if __name__ == "__main__":
    print("Starting Australian ID Extractor API on http://0.0.0.0:5000")
    print("Health check: http://0.0.0.0:5000/health")
    app.run(host="0.0.0.0", port=5000, debug=False)
