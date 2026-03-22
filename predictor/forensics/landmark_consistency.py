import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

def compute_landmark_consistency(face_img_rgb):
    img = face_img_rgb.astype("uint8")
    results = mp_face_mesh.process(img)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img.shape
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
    left_eye = pts[33]
    right_eye = pts[263]
    nose = pts[1]
    dist_left = np.linalg.norm(left_eye - nose)
    dist_right = np.linalg.norm(right_eye - nose)
    symmetry_diff = abs(dist_left - dist_right)
    return symmetry_diff


def interpret_landmark(score):
    if score < 5:
        return "Facial geometry appears natural"
    elif score < 15:
        return "Minor facial symmetry inconsistencies"
    else:
        return "Strong facial landmark inconsistencies detected"


def analyse_landmark_consistency(face_img_rgb):
    score = compute_landmark_consistency(face_img_rgb)
    if score is None:
        return {
            "landmark_score": None,
            "landmark_interpretation": "No face landmarks detected"
        }
    interpretation = interpret_landmark(score)
    return {
        "landmark_score": round(score,2),
        "landmark_interpretation": interpretation
    }