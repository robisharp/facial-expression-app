# facial-expression-app
@"
# 😊 FaceRead — Facial Expression Recognition AI

A deep learning project that detects **7 facial expressions** in real-time using a CNN trained on the FER-2013 dataset.

🌐 **Live Demo:** https://robisharp.github.io/facial-expression-app/
🔧 **Backend API:** https://facial-expression-app-ekkd.onrender.com

## 🎯 What it does
- Detects faces in any uploaded image or live webcam feed
- Predicts one of 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Draws bounding boxes with confidence scores
- Works on any device — mobile, tablet, desktop

## 🧠 Model
- Dataset: FER-2013 (28,709 training images)
- Architecture: VGG-style CNN (6M parameters)
- Validation Accuracy: ~59% (10 epochs, CPU)
- Framework: TensorFlow/Keras

## 🛠️ Tech Stack
- Deep Learning: TensorFlow / Keras
- Face Detection: OpenCV Haar Cascade
- Backend: FastAPI + Uvicorn
- Frontend: Pure HTML / CSS / JS
- Hosting: Render (backend) + GitHub Pages (frontend)

## 🚀 Run Locally
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then open index.html in your browser.

## 👨‍💻 Author
Robi Sharp — github.com/robisharp

⭐ If you found this useful, please give it a star!
"@ | Set-Content "C:\Users\robis\OneDrive\Documents\facial-expression-app\README.md"
