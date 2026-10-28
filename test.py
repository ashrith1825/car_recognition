import argparse
from pathlib import Path

from inference import MODEL_PATH, decode_image, load_model, predict_image


def predict(image_path: Path, checkpoint_path: Path) -> tuple[str, float]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        image = decode_image(image_path.read_bytes())
    except OSError as error:
        raise ValueError(f"Could not read a valid image from {image_path}") from error

    model, class_names = load_model(checkpoint_path)
    return predict_image(image, model, class_names)


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
