import json
import torch
import timm
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file
from .metadata_analyzer import analyze_metadata
from .forensics_analyzer import analyze_forensics
from .scoring import score_from_evidence, label


MODEL_DIR = "kaan_model"


# -----------------------------
# Load model configuration
# -----------------------------
with open(f"{MODEL_DIR}/config.json", "r") as f:
    config = json.load(f)

TEMPERATURE = config.get("temperature", 1.0)


# -----------------------------
# Load Kaan AI detector
# -----------------------------
model = timm.create_model(
    "vit_base_patch16_clip_224.openai",
    pretrained=False,
    num_classes=1,
    img_size=256
)

weights = load_file(f"{MODEL_DIR}/model.safetensors")
model.load_state_dict(weights)

model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


# -----------------------------
# Main detection function
# -----------------------------
def detect_ai_image(image_path):

    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            logit = model(image).squeeze().item()

        # Temperature calibration
        calibrated_logit = logit / TEMPERATURE

        p_real = torch.sigmoid(
            torch.tensor(calibrated_logit)
        ).item()

        p_ai = 1 - p_real

        # -----------------------------
        # Decision threshold
        # -----------------------------
        if p_real >= 0.60:
            prediction = "REAL"

        elif p_real <= 0.40:
            prediction = "AI-GENERATED"

        else:
            prediction = "UNCERTAIN"

        return {
            "prediction": prediction,
            "real_probability": round(p_real * 100, 2),
            "ai_probability": round(p_ai * 100, 2),
            "confidence": round(max(p_real, p_ai) * 100, 2)
        }

    except Exception as e:

        return {
            "prediction": "ERROR",
            "real_probability": 0,
            "ai_probability": 0,
            "confidence": 0,
            "error": str(e)
        }

def analyze_image(image_path):

    # --------------------------------
    # 1. AI detector
    # --------------------------------
    ai_result = detect_ai_image(image_path)

    # --------------------------------
    # 2. Metadata analysis
    # --------------------------------
    metadata_result = analyze_metadata(image_path)

    # --------------------------------
    # 3. Pixel-level forensics
    # --------------------------------
    forensic_result = analyze_forensics(image_path)

    prediction = ai_result["prediction"]
    real_probability = ai_result["real_probability"]
    ai_probability = ai_result["ai_probability"]

    # --------------------------------
    # 4. Build evidence
    # --------------------------------
    evidence = [
        {
            "name": "AI Image Detector",
            "risk": ai_probability,
            "weight": 1.0,
            "detail": (
                f"The AI detector estimates a "
                f"{ai_probability:.2f}% probability that "
                f"this image is AI-generated and "
                f"{real_probability:.2f}% probability that "
                f"it is real."
            )
        },

        {
            "name": "Metadata Analysis",
            "risk": metadata_result["risk"],
            "weight": metadata_result["weight"],
            "detail": metadata_result["detail"]
        },

        {
            "name": "Pixel-Level Forensics",
            "risk": forensic_result["risk"],
            "weight": forensic_result["weight"],
            "detail": forensic_result["detail"]
        }
    ]

    # --------------------------------
    # 5. Calculate Trust Score
    # --------------------------------
    trust_score = score_from_evidence(evidence)

    # --------------------------------
    # 6. Overall label
    # --------------------------------
    result_label = label(trust_score)

    # --------------------------------
    # 7. Final result
    # --------------------------------
    return {
        "type": "image",

        "trust_score": trust_score,

        "label": result_label,

        "summary": (
            "DeepVerify analyzed the image using "
            "AI detection, metadata analysis, "
            "and pixel-level forensic analysis."
        ),

        "prediction": prediction,

        "real_probability": real_probability,

        "ai_probability": ai_probability,

        "confidence": ai_result["confidence"],

        "metadata": metadata_result,

        "forensics": forensic_result,

        "evidence": evidence
    }