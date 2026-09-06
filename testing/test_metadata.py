from analyzers.metadata_analyzer import analyze_metadata


result = analyze_metadata("uploads/real3.jpg")

print("\n==============================")
print("       METADATA ANALYSIS")
print("==============================")

print("Status:", result["status"])
print("Detail:", result["detail"])

print("\nCamera Make:", result["camera_make"])
print("Camera Model:", result["camera_model"])
print("Date Taken:", result["date_taken"])
print("Software:", result["software"])
print("GPS Present:", result["gps"])

print("\nFile Information:")
print(result["file_info"])