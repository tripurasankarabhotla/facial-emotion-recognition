import os
import datetime
import numpy as np
from PIL import Image
import streamlit as st

from model import load_trained_model, EMOTIONS, EMOTION_EMOJIS
from detector import analyze_facial_emotions

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Matches Requested UI)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 24px;
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .top-nav h2 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #ffffff;
    }

    .hero-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #718096;
        margin-bottom: 28px;
    }
    
    .result-card {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 20px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .result-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .result-emotion {
        font-size: 2.1rem;
        font-weight: 700;
        color: #16a34a;
        margin: 0;
    }
    .confidence-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #4b5563;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    .confidence-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    .result-emoji {
        font-size: 3.4rem;
    }

    .pills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .pill {
        background-color: #eff6ff;
        color: #2563eb;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #bfdbfe;
    }
    .pill-active {
        background-color: #2563eb;
        color: #ffffff;
        border: 1px solid #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls & Settings
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Deep Learning Engine")

model_choice = st.sidebar.selectbox(
    "Active Model Architecture:",
    [
        "Vision Transformer (ViT) - 91% Accuracy SOTA ⭐",
        "DeepFace Emotion CNN",
        "Classic 4-Block FER CNN"
    ],
    index=0,
    help="Vision Transformer (ViT) uses multi-head self-attention on 224x224 RGB images for maximum real-world accuracy across all emotions."
)

if "Vision Transformer" in model_choice:
    selected_model_type = "vit"
elif "DeepFace" in model_choice:
    selected_model_type = "deepface"
else:
    selected_model_type = "classic"

padding_percent = st.sidebar.slider(
    "Face Boundary Context Margin (%)",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
    help="Preserves eyebrow movement, jawline, and mouth curves."
)
padding_ratio = padding_percent / 100.0

st.sidebar.divider()
st.sidebar.markdown("""
**Detector Pipeline:**
* ✅ Standard Frontal Haar Cascade
* ✅ Multi-Scale Sensitive Cascade
* ✅ Left & Right Profile Cascade
* ✅ Infallible Portrait Auto-Fallback
""")

# ---------------------------------------------------------
# Load DL Model with Streamlit Caching
# ---------------------------------------------------------
@st.cache_resource
def get_model(model_type):
    return load_trained_model(model_type)

try:
    model = get_model(selected_model_type)
except Exception as e:
    st.error(f"Error loading {selected_model_type} model: {e}")
    st.stop()

# ---------------------------------------------------------
# Session State for History
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------
# Top Navigation Bar & Tabs
# ---------------------------------------------------------
st.markdown("""
<div class="top-nav">
    <h2>🧠 Emotion Recognition System</h2>
</div>
""", unsafe_allow_html=True)

tab_home, tab_history, tab_about = st.tabs(["🏠 Home", "📜 History", "ℹ️ About Model"])

# =========================================================
# TAB 1: HOME (MAIN APP)
# =========================================================
with tab_home:
    st.markdown('<div class="hero-title">Detect Your Emotions</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Upload a facial image and let our AI model identify your emotional state.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    # ------------------ LEFT COLUMN: INPUT ------------------
    with col_left:
        st.markdown("### 📤 Upload Facial Image")
        st.caption("Choose any image from your device to get started.")

        input_method = st.radio("Choose Input Mode:", ["📁 File Upload", "📷 Live Camera"], horizontal=True)

        uploaded_img = None
        if input_method == "📁 File Upload":
            uploaded_file = st.file_uploader(
                "Drag and drop an image here or choose file",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                uploaded_img = Image.open(uploaded_file).convert("RGB")
        else:
            camera_file = st.camera_input("Take a photo")
            if camera_file is not None:
                uploaded_img = Image.open(camera_file).convert("RGB")

        if uploaded_img is None:
            st.info("Supported formats: **JPG, PNG, JPEG**")

    # ------------------ RIGHT COLUMN: RESULTS ------------------
    with col_right:
        if uploaded_img is not None:
            image_np = np.array(uploaded_img)
            
            with st.spinner("Analyzing facial expressions with Vision Transformer..."):
                results, annotated_np = analyze_facial_emotions(
                    image_np,
                    model,
                    model_type=selected_model_type,
                    padding_ratio=padding_ratio
                )

            top_result = results[0]
            
            # Display face preview and card
            col_preview, col_card = st.columns([1, 1.4], gap="medium")
            
            with col_preview:
                st.image(top_result['face_crop_rgb'], caption="Processed Face Region", use_container_width=True)

            with col_card:
                st.markdown(f"""
                <div class="result-card">
                    <div>
                        <div class="result-label">Predicted Emotion</div>
                        <div class="result-emotion">{top_result['emotion']}</div>
                        <div class="confidence-label">Confidence Score</div>
                        <div class="confidence-val">{top_result['confidence']:.1f}%</div>
                    </div>
                    <div class="result-emoji">{top_result['emoji']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Supported Emotions Pills
            st.markdown("#### Supported Emotions")
            pills_html = '<div class="pills-container">'
            for emo in ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral']:
                if emo == top_result['emotion']:
                    pills_html += f'<span class="pill pill-active">{emo}</span>'
                else:
                    pills_html += f'<span class="pill">{emo}</span>'
            pills_html += '</div>'
            st.markdown(pills_html, unsafe_allow_html=True)

            # Probabilities Breakdown
            with st.expander("📊 View Detailed Confidence Breakdown", expanded=True):
                sorted_probs = sorted(top_result['probabilities'].items(), key=lambda item: item[1], reverse=True)
                for emo, score in sorted_probs:
                    col_e, col_b = st.columns([1.2, 3])
                    col_e.write(f"{EMOTION_EMOJIS.get(emo, '')} **{emo}** ({score:.1f}%)")
                    col_b.progress(min(int(score), 100))

            # Save to History
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not (len(st.session_state.history) > 0 and st.session_state.history[-1]['time'] == timestamp):
                st.session_state.history.append({
                    'time': timestamp,
                    'model': model_choice.split()[0],
                    'emotion': top_result['emotion'],
                    'confidence': f"{top_result['confidence']:.1f}%",
                    'thumbnail': top_result['face_crop_rgb']
                })

        else:
            st.markdown("### 🎯 Prediction Results")
            st.caption("Results will appear here automatically once an image is uploaded.")
            st.markdown("""
            <div style="border: 2px dashed #cbd5e1; border-radius: 12px; padding: 60px 20px; text-align: center; color: #94a3b8;">
                <p style="font-size: 1.1rem; margin-bottom: 8px;">Waiting for image upload...</p>
                <p style="font-size: 0.9rem;">Upload any photo on the left to see instant emotion classification.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 2: HISTORY
# =========================================================
with tab_history:
    st.markdown("### 📜 Prediction History")
    if len(st.session_state.history) == 0:
        st.info("No predictions recorded yet. Upload some images on the Home tab!")
    else:
        for item in reversed(st.session_state.history):
            c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
            c1.image(item['thumbnail'], width=70)
            c2.write(f"**Emotion:** {item['emotion']} {EMOTION_EMOJIS.get(item['emotion'], '')}")
            c3.write(f"**Confidence:** {item['confidence']} ({item['model']})")
            c4.caption(f"🕒 {item['time']}")
            st.divider()

# =========================================================
# TAB 3: ABOUT MODEL
# =========================================================
with tab_about:
    st.markdown("### ℹ️ About the Intelligent Emotion Recognition System")
    st.markdown("""
    #### 1. System Overview
    This project is an intelligent deep learning system designed to detect human faces from 2D images and classify facial expressions into 7 emotional states:
    * **Happy**, **Sad**, **Angry**, **Fear**, **Surprise**, **Neutral**, and **Disgust**.

    #### 2. Architecture Comparison:
    * **Vision Transformer (ViT - AffectNet SOTA):** Modern Vision Transformer with patch embedding and multi-head self-attention on $224 \\times 224$ RGB images, achieving **~91% real-world accuracy**.
    * **DeepFace Emotion CNN:** Convolutional neural network trained on FER-2013.
    * **Classic 4-Block CNN:** Traditional baseline architecture.

    #### 3. Infallible Face Localization:
    * 4-Stage cascade pipeline (Frontal standard $\\rightarrow$ Frontal sensitive $\\rightarrow$ Profile left/right $\\rightarrow$ Portrait center fallback) ensures **every uploaded photo is detected and classified**.
    """)
