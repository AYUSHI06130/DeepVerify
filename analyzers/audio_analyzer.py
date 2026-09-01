from pathlib import Path
try:
    import librosa
except Exception:
    librosa = None
from .scoring import score_from_evidence, label

def analyze_audio(path: Path):
    evidence = []
    if librosa is None:
        evidence.append({
            "name":"Audio feature extraction unavailable","risk":10,"weight":0.2,
            "detail":"Install librosa to enable the audio baseline."
        })
    else:
        y, sr = librosa.load(str(path), sr=None, mono=True, duration=30)
        if len(y) == 0:
            raise ValueError("Audio file contains no readable samples.")
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        dynamic = float(rms.max() - rms.min())
        zcr_mean = float(zcr.mean())
        risk = 25 if dynamic < 0.01 else 10
        evidence.append({
            "name":"Audio signal profile","risk":risk,"weight":0.45,
            "detail":f"RMS dynamic range={dynamic:.4f}; mean zero-crossing rate={zcr_mean:.4f}. These are indicators only."
        })

    evidence.append({
        "name":"Synthetic-audio classifier not installed","risk":0,"weight":0.9,
        "detail":"This prototype intentionally does not pretend simple signal statistics can reliably prove AI-generated speech."
    })
    trust = score_from_evidence(evidence)
    return {
        "type":"audio","trust_score":trust,"label":label(trust),
        "summary":"Audio is currently a transparent baseline. Add a trained anti-spoofing model for production use.",
        "evidence":evidence
    }