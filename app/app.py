from __future__ import annotations

import os
from flask import Flask, request, jsonify, render_template

from lyric_karaoke.inference.predictor import KaraokePredictor

app = Flask(__name__)

# Prevent abuse
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

UPLOAD_DIR = os.environ.get("KARAOKE_UPLOAD_DIR", "tmp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("KARAOKE_MODEL_PATH", os.path.join("models", "harmonix_lr.pkl"))

_predictor: KaraokePredictor | None = None


def get_predictor() -> KaraokePredictor | None:
    """Lazy-load predictor so the app can run without bundled model weights."""
    global _predictor
    if _predictor is not None:
        return _predictor

    if not os.path.exists(MODEL_PATH):
        return None

    _predictor = KaraokePredictor(
        model_path=MODEL_PATH,
        frame_duration=0.5,
        thr_segments=0.40,
        thr_start=0.35,
        debug=False,
    )
    return _predictor


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large (max 100MB)"}), 413


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    predictor = get_predictor()
    if predictor is None:
        return (
            jsonify(
                {
                    "error": (
                        "No model weights found. Train a model locally or set KARAOKE_MODEL_PATH "
                        "to an existing .pkl."
                    )
                }
            ),
            400,
        )

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)

    try:
        result = predictor.predict(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(debug=True)
