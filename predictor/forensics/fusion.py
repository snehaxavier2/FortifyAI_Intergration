import numpy as np


def normalize(value, min_v, max_v):
    value = max(min_v, min(float(value), max_v))
    denom = max_v - min_v
    if denom == 0:
        return 0.0
    return (value - min_v) / denom

def compute_forensic_score(
    model_conf,
    ela_score,
    noise_score,
    prnu_score,
    upsample_score,
    landmark_score,
    texture_score,
    metadata=None,
    ai_prediction=-1
):
    # Normalize signals
    ela_norm = normalize(ela_score, 0, 40)
    noise_norm = normalize(noise_score, 0, 10)
    prnu_norm = normalize(prnu_score, 0, 0.01)
    upsample_norm = normalize(upsample_score, 0, 10)
    landmark_norm = normalize(landmark_score, 0, 100)
    texture_norm = normalize(texture_score, 0, 100)
    if ai_prediction == 0:
        model_norm = 1.0 - (model_conf / 100.0)
    elif ai_prediction == 1:
        model_norm = model_conf / 100.0
    else:
        model_norm = model_conf / 100.0

    forensic_ensemble = (
        0.15 * ela_norm      +
        0.12 * noise_norm    +
        0.08 * prnu_norm     +
        0.25 * upsample_norm +
        0.20 * landmark_norm +
        0.20 * texture_norm
    )
    if ai_prediction == 0 and forensic_ensemble > 0.38:
        model_weight     = 0.25
        forensic_weight  = 0.75
    else:
        model_weight     = 0.55
        forensic_weight  = 0.45

    forensic_score = (
        model_weight    * model_norm      +
        forensic_weight * forensic_ensemble
    )
    has_camera_metadata = (
        metadata and isinstance(metadata, dict) and
        any(tag in metadata for tag in {"Make", "Model", "GPSInfo", "DateTimeOriginal"})
    )
    noise_elevated     = noise_norm    > 0.38
    prnu_elevated      = prnu_norm     > 0.018
    landmark_elevated  = landmark_norm > 0.65
    upsample_elevated  = upsample_norm > 0.25   
    gan_signals = sum([noise_elevated, prnu_elevated, 
                       landmark_elevated, upsample_elevated])
    if not has_camera_metadata and gan_signals >= 2:
        gan_strength = (
            0.40 * noise_norm +
            0.35 * prnu_norm  +
            0.25 * texture_norm
        )
        gan_floor = 0.35 + (0.20 * gan_strength)
        forensic_score = max(forensic_score, gan_floor)
    elif not has_camera_metadata and prnu_elevated and noise_elevated:
        gan_floor = 0.33 + (0.15 * noise_norm)
        forensic_score = max(forensic_score, gan_floor)
    
    if metadata and isinstance(metadata, dict):
        if "Make" in metadata or "Model" in metadata:
            forensic_score -= 0.08
        if "GPSInfo" in metadata:
            forensic_score -= 0.05
        camera_tags = {"Make", "Model", "GPSInfo", "DateTimeOriginal", "FocalLength"}
        if not any(tag in metadata for tag in camera_tags):
            forensic_score += 0.06
    forensic_score = max(0.0, min(1.0, forensic_score))   
    return round(forensic_score * 100, 2)     


def final_forensic_decision(score):
    if score > 75:
        label       = "FAKE"
        strength    = "Very High"
        reliability = "HIGH"
        confidence  = round(score, 2)

    elif score > 60:
        label       = "FAKE"
        strength    = "High"
        reliability = "HIGH"
        confidence  = round(score, 2)

    elif score > 42:
        label       = "FAKE"
        strength    = "Moderate"
        reliability = "MEDIUM"
        confidence  = round(score, 2)

    elif score > 30:
        label       = "AUTHENTIC"
        strength    = "Moderate"
        reliability = "MEDIUM"
        confidence  = round(100 - score, 2)

    else:
        label       = "AUTHENTIC"
        strength    = "High"
        reliability = "HIGH"
        confidence  = round(100 - score, 2)

    return label, confidence, reliability, strength