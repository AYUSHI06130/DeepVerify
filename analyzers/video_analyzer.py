from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .image_analyzer import model, transform, TEMPERATURE
from .scoring import score_from_evidence, label


def analyze_video(path: Path):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    duration = count / fps if fps else 0

    # Sample up to 8 frames
    n = min(8, max(1, count))

    indices = np.linspace(
        0,
        max(count - 1, 0),
        n
    ).astype(int)

    frames = []

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ok, frame = cap.read()

        if not ok:
            continue

        # OpenCV BGR → RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy image → PIL
        image = Image.fromarray(rgb)

        # Apply Kaan preprocessing
        image = transform(image).unsqueeze(0)

        # Kaan inference
        with torch.no_grad():
            logit = model(image).squeeze().item()

        # Temperature calibration
        calibrated_logit = logit / TEMPERATURE

        p_real = torch.sigmoid(
            torch.tensor(calibrated_logit)
        ).item()

        p_ai = 1 - p_real

        frames.append({
            "frame": int(idx),
            "time": round(idx / fps, 2),
            "real_probability": round(p_real * 100, 2),
            "ai_probability": round(p_ai * 100, 2)
        })

    cap.release()

    if not frames:
        raise ValueError("No readable frames were found.")

    # Average AI probability across sampled frames
    avg_ai = float(
        np.mean([
            x["ai_probability"]
            for x in frames
        ])
    )

    # Frames where AI probability >= 60%
    suspicious = [
        x for x in frames
        if x["ai_probability"] >= 60
    ]

    # -----------------------------
    # Evidence
    # -----------------------------

    evidence = [
        {
            "name": "Frame-level AI analysis",
            "risk": avg_ai,
            "weight": 0.75,
            "detail": (
                f"Average AI probability across "
                f"{len(frames)} sampled frames: "
                f"{avg_ai:.1f}%."
            )
        }
    ]

    if suspicious:

        evidence.append({
            "name": "Suspicious frames",
            "risk": min(
                90,
                30 + 8 * len(suspicious)
            ),
            "weight": 0.55,
            "detail": (
                f"{len(suspicious)} sampled "
                f"frame(s) had AI probability "
                f"of at least 60%."
            )
        })

    else:

        evidence.append({
            "name": "No strongly suspicious sampled frames",
            "risk": 5,
            "weight": 0.25,
            "detail": (
                "No sampled frame exceeded "
                "the 60% AI threshold. "
                "This does not prove the video "
                "is authentic."
            )
        })

    # Short video warning
    if duration < 2:

        evidence.append({
            "name": "Very short video",
            "risk": 10,
            "weight": 0.2,
            "detail": (
                f"Duration is {duration:.2f}s; "
                "temporal analysis is limited."
            )
        })

    # Calculate DeepVerify trust score
    trust = score_from_evidence(evidence)

    return {
        "type": "video",

        "trust_score": trust,

        "label": label(trust),

        "summary": (
            "Prototype video analysis samples "
            "frames using the Kaan AI image detector. "
            "It does not yet perform true temporal "
            "or lip-sync forensics."
        ),

        "video_info": {
            "fps": round(fps, 2),
            "frames": count,
            "duration": round(duration, 2)
        },

        "frame_results": frames,

        "evidence": evidence
    }