# Intelligent Facial Emotion Recognition System
**Domain:** Deep Learning & Computer Vision  
**Platform:** Python, TensorFlow, OpenCV, Streamlit

---

## 📁 Project Structure
```
emotion_recognition_system/
│
├── app.py                  # Main Web Application (Streamlit UI matching the design)
├── model.py                # Deep CNN Model Architecture and Weights Loader
├── detector.py             # Face Detection (Haar Cascade) & Preprocessing Pipeline
├── realtime_webcam.py      # Real-time Webcam Face Tracking & Emotion Predictor
├── fer.h5                  # Pre-trained Deep Learning Model Weights (FER-2013)
├── requirements.txt        # Project Dependencies
└── README.md               # Documentation & Setup Guide
```

---

## 🚀 How to Run in VS Code

### Step 1: Open the Project in VS Code
1. Open **VS Code**.
2. Go to **File** $\rightarrow$ **Open Folder...**
3. Select this folder:
   `C:\Users\user\.gemini\antigravity\scratch\emotion_recognition_system`

### Step 2: Open the Terminal in VS Code
* Press `Ctrl + ` `~` (or go to **Terminal** $\rightarrow$ **New Terminal** in the top menu).

### Step 3: Run the Web Dashboard
Execute the following command in the VS Code terminal:
```bash
streamlit run app.py
```
* Your browser will automatically open with the web application (`http://localhost:8501`).
* Drag and drop any facial photo (or use the camera option) to see real-time emotion classification with confidence scores!

---

## 🎥 Optional: Run Real-Time Webcam Detection
If you want to demonstrate live video emotion tracking during your viva/project review:
```bash
python realtime_webcam.py
```
* Press **'q'** on your keyboard to close the webcam window.

---

## 🧠 Model & Dataset Summary (For Viva & Reports)
* **Dataset:** FER-2013 (~35,000 grayscale facial images of 48×48 resolution).
* **Classes (7 Emotions):** Happy, Sad, Angry, Fear, Surprise, Neutral, Disgust.
* **Pipeline:**
  1. *Face Detection:* OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
  2. *Preprocessing:* Grayscale conversion $\rightarrow$ 48×48 resizing $\rightarrow$ Normalization $[0, 1]$.
  3. *Inference:* 4-block Convolutional Neural Network with Batch Normalization and Dropout layers $\rightarrow$ Softmax probability distribution.
