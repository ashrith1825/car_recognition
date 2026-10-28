from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


MODEL_PATH = Path(__file__).resolve().parent / "backend" / "model.pth"
DEVICE = torch.device("cpu")

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


def load_model(checkpoint_path: Path = MODEL_PATH) -> tuple[nn.Module, list[str]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("The model checkpoint has an unsupported format.")

    class_to_idx = checkpoint.get("class_to_idx")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(class_to_idx, dict) or not isinstance(state_dict, dict):
        raise RuntimeError("The model checkpoint is missing its weights or class mapping.")

    indices = list(class_to_idx.values())
    if (
        not class_to_idx
        or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
        or sorted(indices) != list(range(len(class_to_idx)))
    ):
        raise RuntimeError("The model checkpoint contains an invalid class mapping.")

    class_names = [""] * len(class_to_idx)
    for class_name, index in class_to_idx.items():
        if not isinstance(class_name, str):
            raise RuntimeError("The model checkpoint contains an invalid class name.")
        class_names[index] = class_name

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(class_names)),
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, class_names


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            uploaded_image.load()
            return uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("This file is not a valid image. Please upload a JPG, PNG, or WebP.") from error


def predict_image(
    image: Image.Image, model: nn.Module, class_names: list[str]
) -> tuple[str, float]:
    image_tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits[0], dim=0)
        confidence, predicted_index = torch.max(probabilities, dim=0)

    index = predicted_index.item()
    if index >= len(class_names):
        raise RuntimeError("The model output does not match the available car classes.")
    return class_names[index], confidence.item()
