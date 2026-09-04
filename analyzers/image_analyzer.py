from pathlib import Path

from PIL import Image, ExifTags
from transformers import pipeline

from .scoring import score_from_evidence, label


# ============================================================
# AI IMAGE DETECTION MODEL
# ============================================================

_MODEL = None

MODEL_NAME = "capcheck/ai-image-detection"


def get_model():
    """
    Load the AI-generated image detector.

    The model is loaded only once and then reused
    for subsequent images.
    """

    global _MODEL

    if _MODEL is None:
        print("Loading AI image detection model...")

        _MODEL = pipeline(
            "image-classification",
            model=MODEL_NAME
        )

        print("AI image detection model loaded.")

    return _MODEL


# ============================================================
# METADATA ANALYSIS
# ============================================================

def metadata_checks(path):
    """
    Check image metadata.

    Missing EXIF is NOT treated as proof of AI generation.
    Many websites and messaging apps remove metadata.
    """

    evidence = []

    try:

        img = Image.open(path)

        exif = img.getexif()

        if not exif:

            evidence.append({
                "name": "Missing EXIF metadata",
                "risk": 12,
                "weight": 0.25,
                "detail": (
                    "No EXIF metadata was found. "
                    "This can happen with AI-generated images, "
                    "but social media platforms and editing software "
                    "also commonly remove metadata."
                )
            })

        else:

            names = [
                ExifTags.TAGS.get(k, str(k))
                for k in exif.keys()
            ]

            evidence.append({
                "name": "EXIF metadata present",
                "risk": 0,
                "weight": 0.15,
                "detail": (
                    "Metadata fields found: "
                    + ", ".join(names[:8])
                )
            })

    except Exception as e:

        evidence.append({
            "name": "Metadata analysis failed",
            "risk": 20,
            "weight": 0.10,
            "detail": str(e)
        })

    return evidence


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(path: Path):

    # --------------------------------------------------------
    # Run AI-generated image detector
    # --------------------------------------------------------

    model = get_model()

    predictions = model(
        str(path),
        top_k=2
    )

    print("Model predictions:")
    print(predictions)


    # --------------------------------------------------------
    # Extract REAL / FAKE probabilities
    # --------------------------------------------------------

    fake_probability = 0.0
    real_probability = 0.0

    raw_predictions = []

    for prediction in predictions:

        label_name = str(prediction["label"])
        score = float(prediction["score"])

        raw_predictions.append({
            "label": label_name,
            "score": round(score, 4)
        })

        normalized_label = label_name.lower()

        if (
            "fake" in normalized_label
            or "ai" in normalized_label
            or "generated" in normalized_label
            or "synthetic" in normalized_label
        ):

            fake_probability = max(
                fake_probability,
                score
            )

        elif (
            "real" in normalized_label
            or "human" in normalized_label
            or "authentic" in normalized_label
        ):

            real_probability = max(
                real_probability,
                score
            )


    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if fake_probability == 0 and real_probability == 0:

        # If model labels are unexpected, use the first
        # prediction rather than pretending we know the answer.

        first = predictions[0]

        first_label = str(first["label"]).lower()
        first_score = float(first["score"])

        if (
            "fake" in first_label
            or "ai" in first_label
            or "generated" in first_label
        ):

            fake_probability = first_score

        else:

            real_probability = first_score


    # --------------------------------------------------------
    # AI detector evidence
    # --------------------------------------------------------

    fake_risk = fake_probability * 100

    if fake_probability >= 0.80:

        detector_detail = (
            f"The AI-image detector estimates a "
            f"{fake_probability * 100:.1f}% probability "
            f"that this image is AI-generated."
        )

    elif fake_probability >= 0.50:

        detector_detail = (
            f"The AI-image detector found moderate evidence "
            f"of synthetic generation "
            f"({fake_probability * 100:.1f}% probability)."
        )

    else:

        detector_detail = (
            f"The AI-image detector estimates a "
            f"{fake_probability * 100:.1f}% probability "
            f"of AI generation."
        )


    evidence = [

        {
            "name": "AI-generated image detector",
            "risk": fake_risk,
            "weight": 0.75,
            "detail": detector_detail
        }

    ]


    # --------------------------------------------------------
    # Metadata checks
    # --------------------------------------------------------

    evidence += metadata_checks(path)


    # --------------------------------------------------------
    # Image resolution check
    # --------------------------------------------------------

    try:

        img = Image.open(path)

        width, height = img.size

        if min(width, height) < 256:

            evidence.append({

                "name": "Very small image",

                "risk": 15,

                "weight": 0.15,

                "detail": (
                    f"Resolution is {width}×{height}. "
                    "Low resolution can reduce forensic confidence."
                )

            })

        else:

            evidence.append({

                "name": "Image resolution",

                "risk": 0,

                "weight": 0.10,

                "detail": (
                    f"Image resolution is {width}×{height}."
                )

            })

    except Exception:

        pass


    # --------------------------------------------------------
    # Calculate final trust score
    # --------------------------------------------------------

    trust = score_from_evidence(evidence)


    # --------------------------------------------------------
    # Create human-readable summary
    # --------------------------------------------------------

    if fake_probability >= 0.80:

        summary = (
            "The image shows strong indicators of AI-generated "
            "or synthetic content. The detector identified a "
            "high probability of AI generation."
        )

    elif fake_probability >= 0.50:

        summary = (
            "The image contains indicators that may be "
            "consistent with AI-generated content. "
            "Additional verification is recommended."
        )

    else:

        summary = (
            "The detector found relatively low evidence of "
            "AI generation. This does not prove that the image "
            "is authentic."
        )


    # --------------------------------------------------------
    # Return result to Flask
    # --------------------------------------------------------

    return {

        "type": "image",

        "trust_score": trust,

        "label": label(trust),

        "summary": summary,

        "model_predictions": raw_predictions,

        "evidence": evidence

    }