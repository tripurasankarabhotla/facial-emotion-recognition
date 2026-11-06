import os
import torch
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

# Standard 7 Emotion categories
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Neutral', 'Disgust']

# Emojis for each emotion
EMOTION_EMOJIS = {
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠',
    'Fear': '😨',
    'Surprise': '😲',
    'Neutral': '😐',
    'Disgust': '🤢'
}

def load_vit_pipeline():
    """
    Loads the State-of-the-Art Vision Transformer (ViT)
    fine-tuned on AffectNet (224x224 RGB, ~91% real-world accuracy).
    Loads offline from local cache.
    """
    model_name = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(model_name, local_files_only=True)
    pipe = pipeline("image-classification", model=model, image_processor=processor, framework="pt")
    return pipe

def build_deepface_model():
    """Builds DeepFace Emotion CNN architecture."""
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])
    return model

def build_classic_cnn_model():
    """Builds classic 4-block FER-2013 CNN."""
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    return model

def load_trained_model(model_type="vit"):
    """
    Loads the requested model architecture:
    - 'vit': Vision Transformer (SOTA ~91% accuracy)
    - 'deepface': DeepFace CNN
    - 'classic': FER-2013 4-Block CNN
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if model_type == "vit":
        return load_vit_pipeline()
    elif model_type == "deepface":
        weights_path = os.path.join(base_dir, "deepface_emotion.h5")
        model = build_deepface_model()
        model.load_weights(weights_path)
        return model
    else:
        weights_path = os.path.join(base_dir, "fer.h5")
        model = build_classic_cnn_model()
        model.load_weights(weights_path)
        return model
