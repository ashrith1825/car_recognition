import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------------
# Initialization
# ----------------------
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing to allow your frontend to communicate with this backend.
CORS(app)

# ----------------------
# Configuration
# ----------------------
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Load Model (once, at startup)
# ----------------------
print(f"🔄 Loading model from {MODEL_PATH} to {DEVICE}...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Recreate the exact model architecture from your training script
    model = models.resnet50(weights=None)
    num_classes = len(checkpoint["class_to_idx"])
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Load the learned weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    # Set the model to evaluation mode
    model.eval()

    # Store class mappings
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("✅ Model loaded successfully!")

except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}.")
    print("Please make sure 'model.pth' is in the same directory as 'app.py'")
    model = None
    idx_to_class = {}

# ----------------------
# Image Preprocessing
# ----------------------
# This transform must be identical to the one used for validation/testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------
# Prediction Function
# ----------------------
def predict(image_bytes):
    """
    Takes image bytes, preprocesses the image, and returns the predicted class and confidence.
    """
    if not model:
        raise RuntimeError("Model is not loaded. Cannot perform prediction.")

    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Apply transformations to the image
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        # Get the top prediction
        confidence, pred_class_idx = torch.max(probs, dim=0)

    # Get the class name from the index
    predicted_class_name = idx_to_class[pred_class_idx.item()]

    return predicted_class_name, confidence.item()

# ----------------------
# API Endpoint
# ----------------------
@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    Handles the file upload and returns the prediction result as JSON.
    """
    if model is None:
        return jsonify({"error": "Model is not loaded, check server logs."}), 500

    # Check if a file was posted
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the user selected a file
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        try:
            # Read the file's content into memory
            img_bytes = file.read()
            # Get prediction
            car_name, confidence = predict(img_bytes)

            print(f"🚗 Prediction: {car_name}, Confidence: {confidence:.2f}")

            # Return the result
            return jsonify({
                "predicted_car": car_name,
                "confidence": round(confidence, 4)
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": "Error processing the image"}), 500

    return jsonify({"error": "An unknown error occurred"}), 500

# ----------------------
# Run the App
# ----------------------
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the server accessible from other devices on your network
    app.run(debug=True, host='0.0.0.0', port=5000)
# Access the app at http://localhost:5000 or http://<your-ip>:5000