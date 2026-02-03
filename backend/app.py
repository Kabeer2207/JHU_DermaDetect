import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# =========================
# App setup
# =========================

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR = PROJECT_ROOT / "models"


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(MODELS_DIR / "resnet50_v3.pth")
)

CLASS_NAMES_PATH = os.environ.get(
    "CLASS_NAMES_PATH",
    str(MODELS_DIR / "class_names.json")
)

# =========================
# Load class names (AUTO)
# =========================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"class_names.json not found at {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

print("✅ Loaded class names:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}: {name}")

# =========================
# Image transforms
# =========================

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Model loading
# =========================

def load_model(num_classes: int):
    model = models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    )
    model.eval()
    return model.to(DEVICE)

model = load_model(NUM_CLASSES)
print(f"🤖 Model loaded from: {MODEL_PATH}")

# =========================
# Prediction helper
# =========================

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = inference_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    prediction = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx])

    all_predictions = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    return prediction, confidence, all_predictions

# =========================
# Routes
# =========================

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_frontend(filename):
    return send_from_directory(FRONTEND_DIR, filename)



@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": True,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        image = None

        # Case 1: JSON with base64 (your frontend)
        if request.is_json:
            data = request.get_json(silent=True)
            if data and "imageData" in data:
                import base64
                from io import BytesIO

                image_data = data["imageData"]

                # Strip data URL prefix if present
                if "," in image_data:
                    image_data = image_data.split(",")[1]

                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))

        # Case 2: multipart/form-data (future-proof)
        elif "image" in request.files:
            image_file = request.files["image"]
            image = Image.open(image_file)

        if image is None:
            return jsonify({"error": "No image provided"}), 400

        prediction, confidence, all_predictions = predict_image(image)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence_percentage": round(confidence * 100, 2),
            "all_predictions": all_predictions
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
