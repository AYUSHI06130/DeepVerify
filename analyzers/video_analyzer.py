from pathlib import Path
import cv2
import numpy as np
from .image_analyzer import get_model
from .scoring import score_from_evidence, label

def analyze_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = count / fps if fps else 0
    n = min(8, max(1, count))
    indices = np.linspace(0, max(count-1,0), n).astype(int)

    model = get_model()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = model(rgb, top_k=2)
        fake = sum(float(p["score"]) for p in preds if "fake" in str(p["label"]).lower())
        frames.append({"frame":int(idx),"time":round(idx/fps,2),
                       "fake_probability":round(fake,4)})
    cap.release()
    if not frames:
        raise ValueError("No readable frames were found.")

    avg = float(np.mean([x["fake_probability"] for x in frames]))
    suspicious = [x for x in frames if x["fake_probability"] >= 0.65]

    evidence = [{
        "name":"Frame-level deepfake analysis","risk":avg*100,"weight":0.75,
        "detail":f"Average fake probability across {len(frames)} sampled frames: {avg*100:.1f}%."
    }]
    if suspicious:
        evidence.append({
            "name":"Suspicious frames","risk":min(90,30+8*len(suspicious)),"weight":0.55,
            "detail":f"{len(suspicious)} sampled frame(s) exceeded the 65% threshold."
        })
    else:
        evidence.append({
            "name":"No strongly suspicious sampled frames","risk":5,"weight":0.25,
            "detail":"No sampled frame exceeded 65%. This does not prove the video is authentic."
        })

    if duration < 2:
        evidence.append({
            "name":"Very short video","risk":10,"weight":0.2,
            "detail":f"Duration is {duration:.2f}s; temporal analysis is limited."
        })

    trust = score_from_evidence(evidence)
    return {
        "type":"video","trust_score":trust,"label":label(trust),
        "summary":"Prototype video analysis samples frames. It does not yet perform true temporal or lip-sync forensics.",
        "video_info":{"fps":round(fps,2),"frames":count,"duration":round(duration,2)},
        "frame_results":frames,"evidence":evidence
    }