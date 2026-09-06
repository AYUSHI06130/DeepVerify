import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

print("Loading model...")
print("This may take a while the first time.\n")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded successfully!\n")


image_path = input("Enter image path: ").strip()

image = Image.open(image_path).convert("RGB")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

fake_probability = torch.sigmoid(outputs.logits).item()
real_probability = 1 - fake_probability

print("\n========== RESULT ==========")
print(f"Fake probability : {fake_probability:.4f}")
print(f"Real probability : {real_probability:.4f}")

if fake_probability >= 0.5:
    print("Verdict           : AI-GENERATED / SYNTHETIC")
else:
    print("Verdict           : REAL / AUTHENTIC")

print("============================")