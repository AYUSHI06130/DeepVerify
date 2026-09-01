from pathlib import Path
from PIL import Image, ExifTags
from transformers import pipeline
from .scoring import score_from_evidence, label

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = pipeline("image-classification",
                          model="shivani1511/deepfake-image-detector")
    return _MODEL

def metadata_checks(path):
    evidence = []
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            evidence.append({
                "name":"Missing EXIF metadata", "risk":12, "weight":0.5,
                "detail":"No EXIF metadata was found. This is not proof of manipulation; platforms often strip metadata."
            })
        else:
            names = [ExifTags.TAGS.get(k, str(k)) for k in exif.keys()]
            evidence.append({
                "name":"EXIF metadata present", "risk":0, "weight":0.3,
                "detail":"Metadata fields found: " + ", ".join(names[:8])
            })
    except Exception as e:
        evidence.append({"name":"Metadata analysis failed","risk":20,"weight":0.2,"detail":str(e)})
    return evidence

def analyze_image(path: Path):
    preds = get_model()(str(path), top_k=2)
    fake = sum(float(p["score"]) for p in preds if "fake" in str(p["label"]).lower())
    raw = [{"label":p["label"],"score":round(float(p["score"]),4)} for p in preds]

    evidence = [{
        "name":"Deepfake image model", "risk":fake*100, "weight":0.65,
        "detail":f"Model estimated fake likelihood at {fake*100:.1f}%."
    }]
    evidence += metadata_checks(path)

    try:
        w, h = Image.open(path).size
        if min(w,h) < 256:
            evidence.append({
                "name":"Very small image","risk":15,"weight":0.25,
                "detail":f"Resolution is {w}×{h}; low resolution reduces forensic confidence."
            })
    except Exception:
        pass

    trust = score_from_evidence(evidence)
    return {
        "type":"image","trust_score":trust,"label":label(trust),
        "summary":"The score combines model output and metadata signals. It is not proof of authenticity.",
        "model_predictions":raw,"evidence":evidence
    }