"""
Facial Expression Recognition - FastAPI Backend
Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import io
import base64
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent so inference.py is importable
sys.path.insert(0, str(Path(__file__).parent))
import inference

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Facial Expression Recognition API",
    description="Detect facial expressions in images using deep learning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "fer_model.h5")


@app.on_event("startup")
async def startup():
    if Path(MODEL_PATH).exists():
        inference.load_model(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠  Model not found at {MODEL_PATH}. Train it first with train_model.py")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Facial Expression Recognition API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": inference._model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image (JPEG / PNG) and receive predicted facial expressions.
    Returns the original image with annotated bounding boxes as a base64 string.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()

    try:
        result = inference.predict(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result["face_count"] == 0:
        # No faces found — return original image with a message overlay
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        annotated_b64 = _pil_to_b64(pil_img)
        return JSONResponse({
            **result,
            "annotated_image": annotated_b64,
            "message": "No faces detected in the image.",
        })

    # Draw bounding boxes + labels on the image
    annotated_b64 = _annotate(image_bytes, result["faces"])
    return JSONResponse({**result, "annotated_image": annotated_b64})


@app.post("/predict-base64")
async def predict_b64(payload: dict):
    """Accept base64-encoded image (from webcam capture etc.)."""
    b64_data = payload.get("image", "")
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image.")

    try:
        result = inference.predict(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    annotated_b64 = _annotate(image_bytes, result["faces"]) if result["face_count"] > 0 else _pil_to_b64(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return JSONResponse({**result, "annotated_image": annotated_b64})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hex_to_bgr(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _annotate(image_bytes: bytes, faces: list) -> str:
    img_array = np.frombuffer(image_bytes, np.uint8)
    img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    for face in faces:
        x, y, w, h = face["bbox"]
        color       = _hex_to_bgr(face["color"])
        label       = f"{face['expression']}  {face['confidence']*100:.0f}%"

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Label background
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - lh - 10), (x + lw + 8, y), color, -1)
        cv2.putText(img, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
