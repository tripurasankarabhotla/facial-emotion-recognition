import cv2
import numpy as np
from model import load_trained_model, EMOTIONS
from detector import preprocess_face, extract_face_crop, CASCADE_PATH

def run_realtime_detection():
    print("[INFO] Loading high-accuracy DeepFace emotion model...")
    model = load_trained_model(model_type="deepface")
    
    print("[INFO] Initializing face detector...")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    print("[INFO] Starting webcam feed... (Press 'q' to quit)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Please check permissions or camera connection.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            # Padded crop for better eyebrow/mouth context
            face_roi = extract_face_crop(gray, (x, y, w, h), padding_ratio=0.15)
            if face_roi.size == 0:
                continue

            processed_face = preprocess_face(face_roi, apply_clahe=True)

            # Predict emotion
            preds = model.predict(processed_face, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            emotion = EMOTIONS[top_idx]
            confidence = preds[top_idx] * 100.0

            # Draw bounding box
            color = (46, 204, 113) if emotion in ['Happy', 'Neutral'] else (52, 152, 219)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label banner
            label = f"{emotion} {confidence:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 10, y), color, -1)
            cv2.putText(
                frame,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Emotion Recognition System - Live Webcam (Press 'q' to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam feed stopped.")

if __name__ == "__main__":
    run_realtime_detection()
