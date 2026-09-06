import json
import torch
import timm
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file


MODEL_DIR = "kaan_model"

# -----------------------------
# Load config
# -----------------------------
with open(f"{MODEL_DIR}/config.json", "r") as f:
    config = json.load(f)

temperature = config.get("temperature", 1.0)

print("Temperature:", temperature)


# -----------------------------
# Load model
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
# Preprocessing
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
# Detection function
# -----------------------------
def detect_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(image).squeeze().item()

    # Temperature calibration
    calibrated_logit = logit / temperature

    p_real = torch.sigmoid(
        torch.tensor(calibrated_logit)
    ).item()

    p_fake = 1 - p_real

    print("\n-----------------------------")
    print("Image:", image_path)
    print("-----------------------------")
    print(f"Raw logit:          {logit:.4f}")
    print(f"Calibrated logit:   {calibrated_logit:.4f}")
    print(f"Real probability:   {p_real * 100:.2f}%")
    print(f"AI probability:     {p_fake * 100:.2f}%")

    if p_real >= 0.60:
        result = "REAL"
    else:
        result = "AI-GENERATED"

    print("Result:", result)

    return {
        "result": result,
        "real_probability": p_real,
        "ai_probability": p_fake
    }


# -----------------------------
# Test image
# -----------------------------
detect_image("uploads/ai2.jpg")