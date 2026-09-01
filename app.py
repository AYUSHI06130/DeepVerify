from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from analyzers.image_analyzer import analyze_image
from analyzers.video_analyzer import analyze_video
from analyzers.audio_analyzer import analyze_audio
from analyzers.claim_analyzer import analyze_claim

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024
ALLOWED = {"png","jpg","jpeg","webp","mp4","mov","avi","wav","mp3","m4a"}

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        try:
            mode = request.form.get("mode")
            if mode == "claim":
                claim = request.form.get("claim", "").strip()
                if not claim:
                    raise ValueError("Enter a claim first.")
                result = analyze_claim(claim)
            else:
                f = request.files.get("file")
                if not f or not f.filename:
                    raise ValueError("Choose a file.")
                if not allowed(f.filename):
                    raise ValueError("Unsupported file type.")
                path = UPLOADS / secure_filename(f.filename)
                f.save(path)
                ext = path.suffix.lower()
                if ext in {".jpg",".jpeg",".png",".webp"}:
                    result = analyze_image(path)
                elif ext in {".mp4",".mov",".avi"}:
                    result = analyze_video(path)
                else:
                    result = analyze_audio(path)
                result["filename"] = path.name
        except Exception as exc:
            error = str(exc)
    return render_template("index.html", result=result, error=error)
if __name__ == "__main__":
    app.run(debug=True)    