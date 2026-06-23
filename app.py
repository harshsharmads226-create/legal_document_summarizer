
import logging
 
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
 
from summarizer import (
    summarize_text,
    extract_text_from_pdf,
    EmptyDocumentError,
    PDFExtractionError,
    SummarizerError,
)
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
 
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
 
MIN_MAX_SENTENCES = 1
MAX_MAX_SENTENCES = 30
DEFAULT_MAX_SENTENCES = 8
 
 
def _parse_max_sentences(raw_value) -> int:
    """Parse and clamp a user-supplied max_sentences value, falling back
    to the default for anything missing or invalid."""
    if raw_value is None:
        return DEFAULT_MAX_SENTENCES
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_SENTENCES
    return max(MIN_MAX_SENTENCES, min(value, MAX_MAX_SENTENCES))
 
 
@app.route("/")
def index():
    return render_template("index.html")
 
 
@app.route("/summarize", methods=["POST"])
def summarize_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    max_sentences = _parse_max_sentences(data.get("max_sentences"))
 
    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400
 
    try:
        result = summarize_text(text, max_sentences=max_sentences)
    except EmptyDocumentError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except SummarizerError as e:
        logger.exception("Summarization failed")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception:
        logger.exception("Unexpected error during summarization")
        return jsonify({"success": False, "error": "Internal server error"}), 500
 
    return jsonify({
        "success": True,
        "summary": result.summary,
        "total_sentences": result.total_sentences,
        "summary_sentences": result.sentences_used,
    })
 
 
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
 
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "Only PDF files are supported"}), 400
 
    max_sentences = _parse_max_sentences(request.form.get("max_sentences"))
 
    try:
        text = extract_text_from_pdf(file)
    except PDFExtractionError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        logger.exception("Unexpected error during PDF extraction")
        return jsonify({"success": False, "error": "Failed to read PDF"}), 400
 
    try:
        result = summarize_text(text, max_sentences=max_sentences)
    except EmptyDocumentError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except SummarizerError as e:
        logger.exception("Summarization failed")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception:
        logger.exception("Unexpected error during summarization")
        return jsonify({"success": False, "error": "Internal server error"}), 500
 
    return jsonify({
        "success": True,
        "summary": result.summary,
        "extracted_text": text,
        "total_sentences": result.total_sentences,
        "summary_sentences": result.sentences_used,
    })
 
 
if __name__ == "__main__":
    app.run(debug=True)