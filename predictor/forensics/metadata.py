from PIL import Image
from PIL.ExifTags import TAGS

FORENSIC_FIELDS = {
            "Make",
            "Model",
            "Software",
            "DateTime",
            "DateTimeOrginal",
            "LensModel",
            "ExifImageWidth",
            "ExifImageHeight",
            "GPSInfo"
            }
def extract_metadata(image:Image.Image)->dict:
    metadata= {}
    try:
        exif_data = image.getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag = TAGS.get(tag, tag)
                if tag in FORENSIC_FIELDS:
                    metadata[tag] = str(value)
    except Exception:
        pass
    return metadata