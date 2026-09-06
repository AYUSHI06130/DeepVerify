from pathlib import Path
from PIL import Image, ExifTags
import os


def analyze_metadata(image_path):

    try:
        image = Image.open(image_path)

        exif = image.getexif()

        metadata = {}

        if exif:
            for tag_id, value in exif.items():
                try:
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    metadata[tag] = str(value)
                except Exception:
                    continue

        # Important metadata fields
        camera_make = metadata.get("Make")
        camera_model = metadata.get("Model")
        date_taken = metadata.get("DateTimeOriginal")
        software = metadata.get("Software")
        gps = metadata.get("GPSInfo")

        # --------------------------------
        # Determine metadata status
        # --------------------------------

        if camera_make or camera_model:

            status = "Camera metadata detected"

            detail = (
                "The image contains camera-related metadata"
                " such as manufacturer/model information."
            )

            risk = 5

        elif exif:

            status = "EXIF metadata detected"

            detail = (
                "The image contains EXIF metadata, but no "
                "camera manufacturer or model was identified."
            )

            risk = 10

        else:

            status = "No EXIF metadata detected"

            detail = (
                "No standard EXIF metadata was found. "
                "This can happen with AI-generated images, "
                "screenshots, social-media downloads, or "
                "images that have been re-encoded."
            )

            risk = 15

        # --------------------------------
        # Software/editing indicator
        # --------------------------------

        software_evidence = None

        if software:

            software_evidence = {
                "name": "Software metadata",
                "risk": 20,
                "weight": 0.25,
                "detail": f"Software metadata detected: {software}."
            }

        # --------------------------------
        # File information
        # --------------------------------

        file_size = os.path.getsize(image_path)

        file_info = {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "file_size_kb": round(file_size / 1024, 2)
        }

        result = {
            "status": status,
            "detail": detail,
            "risk": risk,
            "weight": 0.35,
            "metadata": metadata,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "date_taken": date_taken,
            "software": software,
            "gps": bool(gps),
            "file_info": file_info
        }

        if software_evidence:
            result["software_evidence"] = software_evidence

        return result

    except Exception as e:

        return {
            "status": "Metadata analysis failed",
            "detail": str(e),
            "risk": 0,
            "weight": 0,
            "metadata": {},
            "file_info": {}
        }