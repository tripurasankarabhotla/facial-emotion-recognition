import os
import cv2
import numpy as np
from PIL import Image
from model import EMOTIONS, EMOTION_EMOJIS

# Directory of current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_cascade(filename):
    """
    Robust cascade loader:
    1. Checks local repository directory
    2. Falls back to cv2.data.haarcascades
    """
    local_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(local_path):
        cascade = cv2.CascadeClassifier(local_path)
        if not cascade.empty():
            return cascade
            
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        cv_path = os.path.join(cv2.data.haarcascades, filename)
        if os.path.exists(cv_path):
            cascade = cv2.CascadeClassifier(cv_path)
            if not cascade.empty():
                return cascade
    return None

frontal_cascade = load_cascade('haarcascade_frontalface_default.xml')
profile_cascade = load_cascade('haarcascade_profileface.xml')

def detect_faces_robust(image_np):
    """
    Multi-stage face detection that GUARANTEES every image is processed:
    1. Frontal cascade (standard scale)
    2. Frontal cascade (sensitive scale)
    3. Profile cascade (left profile)
    4. Profile cascade (right profile via flip)
    5. Center fallback (if all else fails, use center 85% of image)
    """
    h_img, w_img = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np

    # Stage 1 & 2: Frontal cascade
    if frontal_cascade is not None and not frontal_cascade.empty():
        faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
        if len(faces) > 0:
            return gray, list(faces), "frontal_standard"

        faces = frontal_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            return gray, list(faces), "frontal_sensitive"

    # Stage 3 & 4: Profile cascade
    if profile_cascade is not None and not profile_cascade.empty():
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            return gray, list(faces), "profile_left"

        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces_flipped) > 0:
            corrected_faces = []
            for (fx, fy, fw, fh) in faces_flipped:
                orig_x = w_img - (fx + fw)
                corrected_faces.append((orig_x, fy, fw, fh))
            return gray, corrected_faces, "profile_right"

    # Stage 5: Infallible fallback (Center crop of photo)
    pad_w = int(w_img * 0.08)
    pad_h = int(h_img * 0.08)
    fallback_box = (pad_w, pad_h, w_img - 2 * pad_w, h_img - 2 * pad_h)
    return gray, [fallback_box], "center_fallback"

def extract_face_crop(image, box, padding_ratio=0.15):
    """
    Extracts face crop with extra padding so eyebrows and mouth are fully preserved.
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = box

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    return image[y1:y2, x1:x2]

def analyze_facial_emotions(image_np, model, model_type="vit", padding_ratio=0.15):
    """
    Runs multi-stage face detection and high-accuracy emotion classification.
    """
    gray, faces, detection_stage = detect_faces_robust(image_np)
    annotated = image_np.copy()
    results = []

    for box in faces:
        x, y, w, h = box
        
        # Padded face crop
        face_rgb_padded = extract_face_crop(image_np, box, padding_ratio)
        if face_rgb_padded.size == 0:
            face_rgb_padded = image_np

        pil_crop = Image.fromarray(face_rgb_padded)

        if model_type == "vit":
            # Vision Transformer inference
            vit_preds = model(pil_crop)
            probs_dict = {}
            for item in vit_preds:
                label_cap = item['label'].capitalize()
                probs_dict[label_cap] = float(item['score'] * 100.0)
            
            # Fill missing standard emotions with small baseline
            for emo in EMOTIONS:
                if emo not in probs_dict:
                    probs_dict[emo] = 0.1

            # Top emotion
            sorted_items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
            top_emotion = sorted_items[0][0]
            top_confidence = sorted_items[0][1]

        else:
            # DeepFace / Classic CNN inference (48x48 grayscale)
            face_gray_padded = cv2.cvtColor(face_rgb_padded, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(face_gray_padded, (48, 48), interpolation=cv2.INTER_AREA)
            normalized = resized.astype("float32") / 255.0
            face_input = np.expand_dims(np.expand_dims(normalized, -1), 0)

            preds = model.predict(face_input, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            top_emotion = EMOTIONS[top_idx]
            top_confidence = float(preds[top_idx] * 100.0)
            probs_dict = {EMOTIONS[i]: float(preds[i] * 100.0) for i in range(len(EMOTIONS))}

        results.append({
            'box': box,
            'emotion': top_emotion,
            'confidence': top_confidence,
            'emoji': EMOTION_EMOJIS.get(top_emotion, '🙂'),
            'probabilities': probs_dict,
            'face_crop_rgb': pil_crop,
            'detection_stage': detection_stage
        })

        # Draw box and emotion tag
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (46, 204, 113), 3)
        label_text = f"{top_emotion} ({top_confidence:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x, y - th - 12), (x + tw + 10, y), (46, 204, 113), -1)
        cv2.putText(annotated, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return results, annotated
