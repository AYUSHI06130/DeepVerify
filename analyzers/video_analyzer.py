from pathlib import Path

import cv2
import numpy as np

from .image_analyzer import detect_ai_image
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

        # OpenCV gives BGR.
        # Convert to RGB for the image detector.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Temporary frame used by the existing Kaan image detector.
        temp_path = path.parent / "_temp_video_frame.jpg"

        cv2.imwrite(
            str(temp_path),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )

        # Run Kaan image detector
        result = detect_ai_image(temp_path)

        if result.get("prediction") == "ERROR":
            try:
                temp_path.unlink()
            except Exception:
                pass

            cap.release()

            raise ValueError(
                result.get(
                    "error",
                    "Image detector failed while analyzing a video frame."
                )
            )

        # IMPORTANT:
        # detect_ai_image() returns ai_probability as a percentage
        # such as 78.56, NOT 0.7856.
        #
        # Convert percentage -> probability here.
        ai_percentage = float(
            result.get("ai_probability", 0)
        )

        ai_probability = ai_percentage / 100.0

        real_probability = 1.0 - ai_probability

        frames.append({
            "frame": int(idx),
            "time": round(idx / fps, 2),

            # Keep this internally as 0-1.
            # index.html multiplies it by 100 for display.
            "fake_probability": round(ai_probability, 4),

            # Useful if we want to display this later.
            "real_probability": round(real_probability, 4)
        })

        # Delete temporary frame
        try:
            temp_path.unlink()
        except Exception:
            pass

    cap.release()

    if not frames:
        raise ValueError("No readable frames were found.")

    # --------------------------------------------------
    # Average AI probability across sampled frames
    # --------------------------------------------------

    avg = float(
        np.mean([
            frame["fake_probability"]
            for frame in frames
        ])
    )

    # --------------------------------------------------
    # Find suspicious frames
    # --------------------------------------------------
    # Threshold is now correctly 0.65 = 65%
    suspicious = [
        frame
        for frame in frames
        if frame["fake_probability"] >= 0.65
    ]

    # --------------------------------------------------
    # Build evidence
    # --------------------------------------------------

    evidence = [
        {
            "name": "Frame-level AI analysis",
            "risk": avg * 100,
            "weight": 0.75,
            "detail": (
                f"Average AI probability across "
                f"{len(frames)} sampled frames: "
                f"{avg * 100:.1f}%."
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
                f"{len(suspicious)} sampled frame(s) "
                f"exceeded the 65% AI-probability threshold."
            )
        })

    else:

        evidence.append({
            "name": "No strongly suspicious sampled frames",
            "risk": 5,
            "weight": 0.25,
            "detail": (
                "No sampled frame exceeded 65%. "
                "This does not prove the video is authentic."
            )
        })

    # --------------------------------------------------
    # Very short video
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Calculate trust score
    # --------------------------------------------------

    trust = score_from_evidence(evidence)

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    return {
        "type": "video",

        "trust_score": trust,

        "label": label(trust),

        "summary": (
            "DeepVerify sampled frames from the video "
            "and analyzed them using the Kaan AI-image "
            "detector. The frame probabilities were "
            "aggregated to estimate the overall AI-generated "
            "visual risk. This prototype does not yet perform "
            "true temporal or lip-sync deepfake analysis."
        ),

        "video_info": {
            "fps": round(fps, 2),
            "frames": count,
            "duration": round(duration, 2)
        },

        "frame_results": frames,

        "evidence": evidence
    }