import os
import cv2
import numpy as np
from datetime import datetime


def compute_upsampling_artifacts(face_img_rgb: np.ndarray):
    face_img_rgb = face_img_rgb.astype("uint8")
    gray = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag_norm = mag_norm.astype("uint8")
    return mag_norm


# detect strong periodic spikes in high frequencies
def compute_upsampling_score(freq_map: np.ndarray):
    
    h, w = freq_map.shape
    mask = np.ones((h, w), dtype=bool)
    mask[h//2-10:h//2+10, w//2-10:w//2+10] = False
    freq_no_dc = freq_map.copy().astype(float)
    freq_no_dc[~mask] = 0
    grid_rows, grid_cols = 8, 8
    block_h = h // grid_rows
    block_w = w // grid_cols
    block_maxima = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            block = freq_no_dc[
                i*block_h:(i+1)*block_h,
                j*block_w:(j+1)*block_w
            ]
            if block.size > 0:
                block_maxima.append(float(np.max(block)))

    block_maxima = np.array(block_maxima)
    if len(block_maxima) == 0:
        return 0.0
    mean_peak   = np.mean(block_maxima)
    std_peak    = np.std(block_maxima)
    top_quarter = np.percentile(block_maxima, 75)
    uniformity = mean_peak / (std_peak + 1e-8)
    max_peak     = np.max(block_maxima)
    high_blocks  = np.sum(block_maxima > 0.70 * max_peak)
    spread_ratio = high_blocks / len(block_maxima)
    score = float(uniformity * spread_ratio * 10)
    return round(min(score, 100.0), 2)


def interpret_upsampling(score: float):
    if score < 3.0:
        return "No significant upsampling artifacts"
    elif score < 8.0:
        return "Minor upsampling variations detected"
    elif score < 15.0:
        return "Possible GAN upsampling artifacts detected"
    else:
        return "Strong GAN upsampling artifacts detected"


def save_upsampling_map(freq_map: np.ndarray):
    output_dir = os.path.join("media", "upsampling_outputs")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"upsampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, freq_map)
    return path.replace("\\", "/")

def analyse_upsampling_artifacts(face_img_rgb: np.ndarray):
    freq_map = compute_upsampling_artifacts(face_img_rgb)
    score = compute_upsampling_score(freq_map)
    interpretation = interpret_upsampling(score)
    artifact_path = save_upsampling_map(freq_map)
    return {
        "upsampling_score": score,
        "upsampling_interpretation": interpretation,
        "upsampling_artifact": artifact_path
    }