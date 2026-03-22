import os
import cv2
import numpy as np
from datetime import datetime

def compute_noise_residual(img_rgb: np.ndarray):
    img_rgb = img_rgb.astype("uint8")
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    residual = cv2.absdiff(gray, denoised)
    high_freq = cv2.Laplacian(residual, cv2.CV_64F)
    high_freq = np.absolute(high_freq)
    high_freq = np.uint8(np.clip(high_freq, 0, 255))
    return high_freq


def compute_noise_score(residual: np.ndarray):
    score = float(np.mean(residual))
    return round(score, 2)


def compute_prnu_score(residual: np.ndarray):
    prnu_strength = float(np.var(residual.astype("float32") / 255.0))
    return round(prnu_strength, 4)


def interpret_noise(score: float):
    if score < 3:
        return "Natural sensor noise pattern detected"
    elif score < 7:
        return "Moderate noise irregularities present"
    else:
        return "Strong synthetic noise artifacts detected"
    

def interpret_prnu(prnu_score: float):
    if prnu_score < 5:
        return "Weak sensor pattern detected (possible synthetic image)"
    elif prnu_score < 20:
        return "Moderate sensor pattern present"
    else:
        return "Strong camera sensor pattern detected"
    

def save_noise_map(residual: np.ndarray):
    output_dir = os.path.join("media", "noise_outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"noise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(output_dir, filename)
    normalized = cv2.normalize(residual, None, 255, cv2.NORM_MINMAX)
    cv2.imwrite(path, normalized)
    return path.replace("\\", "/")

def analyse_sensor_noise(face_img_rgb: np.ndarray):
    residual = compute_noise_residual(face_img_rgb)
    noise_score = compute_noise_score(residual)
    noise_interpretation = interpret_noise(noise_score)
    prnu_score = compute_prnu_score(residual)
    prnu_interpretation = interpret_prnu(prnu_score)
    noise_map_path = save_noise_map(residual)
    return{
        "noise_score": noise_score,
        "noise_interpretation": noise_interpretation,
        "prnu_score": prnu_score,
        "prnu_interpretation": prnu_interpretation,
        "noise_artifact": noise_map_path
    }
