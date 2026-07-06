import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ----------------------
# Config
# ----------------------
MODEL_PATH = "backend/model.pth"
DEFAULT_IMAGE_PATH = "backend/data/image.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Image Preprocessing
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------
# Load Model
# ----------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

num_classes = len(checkpoint["class_to_idx"])
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, num_classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# ----------------------
# Predict Function
# ----------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred_class = torch.max(probs, dim=0)

    print(f"\nImage: {image_path}")
    print(f"Predicted Car: {idx_to_class[pred_class.item()]} "
          f"(Confidence: {conf.item():.2f})")

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    predict(image_path)