# DermaDetect
this is a test
DermaDetect is an AI-powered skin disease detection system built using deep learning.
It uses a ResNet50-based convolutional neural network served through a Flask backend
with a modern web-based frontend for image upload, analysis, and diagnosis display.

This project is intended for academic and demonstrational purposes only and is NOT
a substitute for professional medical diagnosis.

---

## Supported Skin Conditions

The current model is trained to classify the following skin conditions:

- Acne Rosacea
- Eczema
- Keratosis
- Milia
- Psoriasis
- Ringworm
- Vitiligo


If the model confidence is below a defined threshold, the system returns "No Disease".


---

## Model Details

Architecture: ResNet50  
Framework: PyTorch  
Input size: 224 x 224 RGB images  
Output: Class probabilities with confidence score  
Mode: Inference-only (training script included separately)

Model weight files (.pth) are intentionally excluded from this repository.

---

## Project Structure

JHU_project/
├── backend/
│   ├── app.py            Flask API for inference
│   ├── train.py          Model training script
├── frontend/
│   ├── index.html        Image upload page
│   ├── page2.html        Analysis/loading page
│   ├── page3.html        Results and diagnosis page
│   ├── style.css         Frontend styling
├── models/
│   └── class_names.json  Model class labels (weights excluded)
├── requirements.txt
├── .gitignore
└── README.md

---

## Setup Instructions (Local)

Step 1: Install dependencies

pip install -r requirements.txt

Step 2: Add model weights

Download the trained model weights and place them in the models directory:

models/
└── resnet50_best.pth

Model weights are not included in this repository.

Step 3: Run the backend server

python backend/app.py

The Flask server will start locally on port 5000 by default.

---

## Frontend Workflow

1. Upload an image on the homepage
2. The image is sent to the backend API for analysis
3. The results page displays:
   - Predicted skin condition
   - Confidence percentage
   - Symptoms and treatment recommendations
   - Medical disclaimer

---

## Medical Disclaimer

This system is provided for educational purposes only.

It is not a medical device.
It must not be used for real-world diagnosis.
Always consult a qualified dermatologist for medical advice.

---

## Future Work

- Docker-based deployment
- Hosting on Hugging Face Spaces
- Dataset expansion and class balancing
- Model optimization and validation improvements

---

## License

This project is licensed under the MIT License.
