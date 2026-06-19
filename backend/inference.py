"""
Facial Expression Recognition - Inference Utility
Loads the trained model and exposes a single predict() function.
"""

import io
import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 48
CLASSES  = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOJIS = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😊",
    "Sad":      "😢",
    "Surprise": "😲",
    "Neutral":  "😐",
}

COLORS = {
    "Angry":    "#E24B4A",
    "Disgust":  "#639922",
    "Fear":     "#7F77DD",
    "Happy":    "#1D9E75",
    "Sad":      "#378ADD",
    "Surprise": "#EF9F27",
    "Neutral":  "#888780",
}

# Haar cascade for face detection
_cascade = None
_model   = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _cascade


def load_model(model_path: str = "fer_model.h5"):
    """Call this once at startup."""
    global _model
    import tensorflow as tf
    _model = tf.keras.models.load_model(model_path)
    return _model


def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Resize, normalise and shape a face crop for inference."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return resized.astype("float32") / 255.0


def predict(image_bytes: bytes) -> dict:
    """
    Given raw image bytes (JPEG/PNG), detect faces and return predictions.

    Returns:
        {
            "faces": [
                {
                    "expression": "Happy",
                    "emoji":      "😊",
                    "color":      "#1D9E75",
                    "confidence": 0.93,
                    "all_scores": {"Angry": 0.01, ...},
                    "bbox":       [x, y, w, h]
                },
                ...
            ],
            "face_count": 1
        }
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Decode bytes → OpenCV array
    img_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for (x, y, w, h) in faces:
        face_crop = img_bgr[y : y + h, x : x + w]
        processed = preprocess_face(face_crop)[np.newaxis, ..., np.newaxis]
        preds      = _model.predict(processed, verbose=0)[0]

        top_idx    = int(np.argmax(preds))
        expression = CLASSES[top_idx]
        all_scores = {cls: round(float(preds[i]), 4) for i, cls in enumerate(CLASSES)}

        results.append({
            "expression": expression,
            "emoji":      EMOJIS[expression],
            "color":      COLORS[expression],
            "confidence": round(float(preds[top_idx]), 4),
            "all_scores": all_scores,
            "bbox":       [int(x), int(y), int(w), int(h)],
        })

    return {"faces": results, "face_count": len(results)}
