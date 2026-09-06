from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def analyze_forensics(image_path):

    try:
        # -----------------------------
        # Load image
        # -----------------------------
        image = Image.open(image_path).convert("RGB")

        rgb = np.array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        height, width = gray.shape

        # -----------------------------
        # 1. Sharpness
        # -----------------------------
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # -----------------------------
        # 2. Noise estimation
        # -----------------------------
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        noise = float(np.std(gray.astype(np.float32) -
                             blurred.astype(np.float32)))

        # -----------------------------
        # 3. Entropy
        # -----------------------------
        histogram = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        histogram = histogram / histogram.sum()

        entropy = float(
            -np.sum(
                histogram[histogram > 0] *
                np.log2(histogram[histogram > 0])
            )
        )

        # -----------------------------
        # 4. Color statistics
        # -----------------------------
        mean_rgb = rgb.mean(axis=(0, 1))

        std_rgb = rgb.std(axis=(0, 1))

        # -----------------------------
        # 5. Aspect ratio
        # -----------------------------
        aspect_ratio = width / height if height else 0

        # -----------------------------
        # Basic forensic interpretation
        # -----------------------------
        indicators = []

        # Extremely low noise
        if noise < 2:
            indicators.append(
                "Very low pixel-level noise detected."
            )

        # Very high sharpness
        if sharpness > 2500:
            indicators.append(
                "Unusually high image sharpness detected."
            )

        # Very low entropy
        if entropy < 5:
            indicators.append(
                "Low grayscale entropy detected."
            )

        # Resolution
        if width == height:
            indicators.append(
                "Image uses a square aspect ratio."
            )

        # These are deliberately weak signals.
        forensic_risk = 0

        if noise < 2:
            forensic_risk += 10

        if sharpness > 2500:
            forensic_risk += 8

        if entropy < 5:
            forensic_risk += 7

        forensic_risk = min(forensic_risk, 25)

        # -----------------------------
        # Result
        # -----------------------------
        if indicators:

            detail = (
                "Pixel-level analysis found the following "
                "characteristics: "
                + " ".join(indicators)
                + " These signals are weak on their own "
                  "and can also occur in genuine images."
            )

        else:

            detail = (
                "No unusually strong pixel-level "
                "forensic indicators were detected. "
                "This does not prove that the image is real."
            )

        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 3),

            "sharpness": round(sharpness, 2),
            "noise_level": round(noise, 2),
            "entropy": round(entropy, 2),

            "mean_rgb": [
                round(float(x), 2)
                for x in mean_rgb
            ],

            "std_rgb": [
                round(float(x), 2)
                for x in std_rgb
            ],

            "indicators": indicators,

            "risk": forensic_risk,

            "weight": 0.30,

            "detail": detail
        }

    except Exception as e:

        return {
            "width": 0,
            "height": 0,
            "aspect_ratio": 0,
            "sharpness": 0,
            "noise_level": 0,
            "entropy": 0,
            "mean_rgb": [],
            "std_rgb": [],
            "indicators": [],
            "risk": 0,
            "weight": 0,
            "detail": f"Forensic analysis failed: {str(e)}"
        }