import os
import datetime
import cv2
import numpy as np
import torch
import base64
import gdown # type: ignore
from io import BytesIO
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .gradcam import predict_with_gradcam
from .model import HybridModel
from .forensics.hashing import generate_sha256
from .forensics.metadata import extract_metadata
from .forensics.custody import log_event
from .forensics.ela import perform_ela, compute_ela_score, interpret_ela, save_ela_image
from .forensics.noise_residual import analyse_sensor_noise
from .forensics.upsampling_artifact import analyse_upsampling_artifacts
from .forensics.landmark_consistency import analyse_landmark_consistency
from .forensics.texture_glcm import analyse_texture_glcm
from .forensics.fusion import compute_forensic_score, final_forensic_decision

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")

# Auto-download weights if missing
if not os.path.exists(MODEL_PATH):
    print("[FortifyAI v5] Downloading model weights...")
    gdown.download(
        "https://drive.google.com/uc?id=1QUqVIArl6BbgEJ-ihUKBs-9Kp7vnYeq2",
        MODEL_PATH,
        quiet=False
    )

# Model loading
def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}\n"
            "Check your Google Drive link or place best_model.pth manually."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(pretrained=False).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[FortifyAI v5] Model loaded: {MODEL_PATH}")
    print(f"[FortifyAI v5] Device: {device}")
    return model

MODEL = _load_model()
# Preprocessing 
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),          
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
DEVICE = torch.device("cuda" if torch.cuda.is_available()else "cpu")
MTCNN_DETECTOR = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    device=DEVICE    
)

def _resize_for_detection(img_rgb, max_size=640):
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_size:
        return img_rgb
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_rgb,(new_w, new_h))
    return resized

def _detect_and_crop_face(img_rgb: np.ndarray):
    h, w = img_rgb.shape[:2]
    detect_img = _resize_for_detection(img_rgb)
    dh, dw = detect_img.shape[:2]
    scale_x = w / dw
    scale_y = h / dh
    boxes, probs = MTCNN_DETECTOR.detect(detect_img)
    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        margin = int(0.15 * (x2-x1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size > 0 and face_crop.shape[0] > 50 and face_crop.shape[1] > 50:
            return face_crop, True
    return img_rgb, False

# Encode input face crop for debugging — confirms what model actually saw

    face_debug_pil = Image.fromarray(face_crop).resize((112, 112))
    buf = BytesIO()
    face_debug_pil.save(buf, format="JPEG", quality=70)
    face_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

def _save_gradcam(overlay: np.ndarray, prefix: str = "gradcam") -> str | None:
    output_dir = getattr(settings, "GRADCAM_OUTPUT_DIR", None)
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath    = os.path.join(output_dir, f"{prefix}_{timestamp}.png")
    cv2.imwrite(filepath, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return filepath


# View 

@api_view(["POST"])
def predict(request):
    image_file = request.FILES.get("image")
    log_event("File uploaded")
    file_bytes = image_file.read()
    file_hash = generate_sha256(file_bytes)
    log_event("SHA256 hash generated")
    image_file.seek(0)
    print(f"[fortifyAI] SHA256:{file_hash}")
    if not image_file:
        return Response({"error": "No image uploaded."}, status=400)

    # Load image
    try:
        image   = Image.open(image_file)
        image = ImageOps.exif_transpose(image)
        metadata = extract_metadata(image)
        log_event("Metadata extracted")
        if not metadata:
            metadata = {"info": "No frorensic metadata available"}
        image = image.convert("RGB")
        img_rgb = np.array(image)
        # Face detection
        face_crop, face_detected = _detect_and_crop_face(img_rgb)
        log_event("Face detection completed")
        face_pil = Image.fromarray(face_crop)
        ela_image = perform_ela(face_pil)
        ela_score = compute_ela_score(ela_image)
        ela_interpretation = interpret_ela(ela_score)
        ela_path = save_ela_image(ela_image)
        log_event("ELA forensics analysis completed")
        print("ELA generated sucessfully")
        sensor_noise = analyse_sensor_noise(face_crop)
        log_event("Noise residual analysis completed")
        print("Noise residual generated successfully")
        upsampling = analyse_upsampling_artifacts(face_crop)
        log_event("Upsampling artifact analysis completed")
        landmark = analyse_landmark_consistency(face_crop)
        log_event("Landmark consistency analysis completed")
        texture = analyse_texture_glcm(face_crop)
        log_event("Texture anomaly analysis completed")


    except Exception as e:
        return Response({"error": f"Invalid image: {str(e)}"}, status=400)



    # Preprocess — 224×224
    tensor = TRANSFORM(face_pil).unsqueeze(0)

    # Inference + Grad-CAM
    try:
        prediction, probability, overlay, cam = predict_with_gradcam(MODEL, tensor)
        print("CAM  shape:",  cam.shape)
        log_event("Deepfake analysis completed")
    except Exception as e:
        return Response({"error": f"Inference failed: {str(e)}"}, status=500)

    # Debug log
    print(f"[FortifyAI v5] prob={probability:.6f} | pred={prediction} | face={face_detected}")

    # Confidence — always reflects certainty in the label returned
    raw_confidence = probability if prediction == 1 else (1 - probability)
    confidence     = round(raw_confidence * 100, 1)
#    if confidence >= 80:
#        detection_reliability = "HIGH" 
#    elif confidence >= 60:
#        detection_reliability = "MEDIUM"
#    else:
#        detection_reliability = "LOW CONFIDENCE"
#    label       = "DEEPFAKE" if prediction == 1 else "AUTHENTIC"
    explanation = (
        "Manipulated facial regions detected via spectral artifact analysis."
        if prediction == 1
        else "No manipulation artifacts detected. Natural facial patterns confirmed."
    )    

    forensic_score = compute_forensic_score(
        confidence,
        ela_score,
        sensor_noise["noise_score"],
        sensor_noise["prnu_score"],
        upsampling["upsampling_score"],
        landmark["landmark_score"],
        texture["texture_score"],
        metadata,
        ai_prediction=prediction
    )
    final_label, forensic_score, reliability = final_forensic_decision(forensic_score)
#    if prediction == 0 and confidence > 70:
#        label = "AUTHENTIC"
#        detection_reliability = "HIGH"
#    else:
    label = final_label
    confidence = round(forensic_score, 2)
    detection_reliability = reliability

    saved_path  = _save_gradcam(overlay)

    response_data = {
        "prediction":    label,
        "confidence":    confidence,
        "detection_reliability":  detection_reliability,
        "sha256_hash":   file_hash,
        "metadata"   :   metadata,
        "explanation":   explanation,
        "gradcam_artifact":   saved_path,
        "ela_score"  :   ela_score,
        "ela_interpretation": ela_interpretation,
        "ela_artifact" : ela_path,
        "sensor_noise":  sensor_noise, 
        "upsampling_analysis": upsampling,
        "landmark_analysis": landmark,
        "texture_analysis": texture,
        "face_detected": face_detected,
    }
    if saved_path:
        response_data["saved_path"] = saved_path

    return Response(response_data)