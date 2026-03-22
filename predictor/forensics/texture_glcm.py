import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def compute_glcm_texture(face_img_rgb):
    gray = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    score = contrast * (1 - homogeneity)
    return score


def interpret_texture(score):
    if score < 5:
        return "Natural texture statistics"
    elif score < 15:
        return "Possible synthetic texture patterns"
    else:
        return "Strong GAN texture anomalies detected"


def analyse_texture_glcm(face_img_rgb):
    score = compute_glcm_texture(face_img_rgb)
    interpretation = interpret_texture(score)
    return {
        "texture_score": round(float(score),2),
        "texture_interpretation": interpretation
    }