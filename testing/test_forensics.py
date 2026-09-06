from analyzers.forensics_analyzer import analyze_forensics


result = analyze_forensics("uploads/real3.jpg")

print("\n==============================")
print("       IMAGE FORENSICS")
print("==============================")

print("Resolution:",
      result["width"],
      "x",
      result["height"])

print("Aspect ratio:",
      result["aspect_ratio"])

print("Sharpness:",
      result["sharpness"])

print("Noise:",
      result["noise_level"])

print("Entropy:",
      result["entropy"])

print("Mean RGB:",
      result["mean_rgb"])

print("\nIndicators:")

if result["indicators"]:
    for indicator in result["indicators"]:
        print("-", indicator)
else:
    print("- None")

print("\nRisk:",
      result["risk"])