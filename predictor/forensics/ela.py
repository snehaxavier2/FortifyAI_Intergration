import io
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from datetime import datetime

def perform_ela(image_pil, quality=90):
    image_pil = image_pil.convert("RGB")
    buffer = io.BytesIO()
    image_pil.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    ela_image = ImageChops.difference(image_pil, recompressed)
    extrema = ela_image.getextrema()
    max_diff = 0
    for channel in extrema:
        max_diff = max(max_diff, channel[1])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / float(max_diff)
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image 


def compute_ela_score(ela_image):
    arr = np.array(ela_image).astype("float32")
    score = float(arr.std())
    return round(score, 2)

def interpret_ela(score):
    if score < 5:
        return "No significant compresiion anomalies detected"
    elif score < 15:
        return "Possible minor editing artifacts detected"
    else:
        return "Compression inconsistencies detected"

def save_ela_image(ela_image):
    output_dir = os.path.join("media", "ela_outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename =f"ela_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.png"
    path = os.path.join(output_dir, filename)
    ela_image.save(path)
    return path.replace("\\", "/")