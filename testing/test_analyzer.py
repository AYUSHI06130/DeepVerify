from analyzers.image_analyzer import detect_ai_image


result = detect_ai_image("uploads/real3.jpg")

print("\n==============================")
print("       DEEPVERIFY RESULT")
print("==============================")

print("Prediction:", result["prediction"])
print("Real:", result["real_probability"], "%")
print("AI:", result["ai_probability"], "%")
print("Confidence:", result["confidence"], "%")