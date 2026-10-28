import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "backend" / "model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_model(
    checkpoint_path: Path,
) -> tuple[nn.Module, dict[int, str]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("The checkpoint has an unsupported format.")

    class_to_idx = checkpoint.get("class_to_idx")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(class_to_idx, dict) or not isinstance(state_dict, dict):
        raise RuntimeError("The checkpoint is missing its weights or class mapping.")

    indices = list(class_to_idx.values())
    if (
        not class_to_idx
        or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
        or sorted(indices) != list(range(len(class_to_idx)))
    ):
        raise RuntimeError("The checkpoint contains an invalid class mapping.")

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(class_to_idx)),
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, {index: name for name, index in class_to_idx.items()}


def predict(image_path: Path, checkpoint_path: Path) -> tuple[str, float]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"Could not read a valid image from {image_path}") from error

    model, idx_to_class = load_model(checkpoint_path)
    image_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits[0], dim=0)
        confidence, predicted_index = probabilities.max(dim=0)

    predicted_class = idx_to_class.get(predicted_index.item())
    if predicted_class is None:
        raise RuntimeError("The model prediction has no matching class name.")
    return predicted_class, confidence.item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the car model shown in an image."
    )
    parser.add_argument("image", type=Path, help="Path to a car image.")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        predicted_class, confidence = predict(args.image, args.checkpoint)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    print(f"Image: {args.image}")
    print(f"Predicted car: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
