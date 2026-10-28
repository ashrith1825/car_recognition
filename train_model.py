import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "backend" / "data"
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "val"
MODEL_PATH = PROJECT_DIR / "backend" / "model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 40
BASE_LEARNING_RATE = 1e-4
HEAD_LEARNING_RATE = 3e-4
MIXUP_ALPHA = 0.4
VALIDATION_FRACTION = 0.1
SEED = 42

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)
EVALUATION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


class TransformedSubset(Dataset):
    """Subset indices with a separate transform for validation images."""

    def __init__(
        self,
        dataset: datasets.ImageFolder,
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.dataset.samples[self.indices[index]]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        return self.transform(image), label


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def stratified_split(
    targets: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        class_indices.setdefault(label, []).append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in class_indices.values():
        rng.shuffle(indices)
        validation_count = (
            max(1, round(len(indices) * validation_fraction))
            if len(indices) > 1
            else 0
        )
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    if not validation_indices:
        raise ValueError(
            "Could not create a validation split. Add at least two training "
            "images for one or more classes, or provide a backend/data/val folder."
        )
    return train_indices, validation_indices


def make_loaders(
    data_dir: Path,
    batch_size: int,
    workers: int,
    validation_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_dir = data_dir / "train"
    validation_dir = data_dir / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training folder not found: {train_dir}. "
            "Organize images as backend/data/train/<class_name>/<image>."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    if len(train_dataset.classes) < 2:
        raise ValueError("The training folder must contain at least two class folders.")
    class_to_idx = train_dataset.class_to_idx

    if validation_dir.is_dir():
        validation_dataset = datasets.ImageFolder(
            validation_dir, transform=EVALUATION_TRANSFORM
        )
        if validation_dataset.class_to_idx != class_to_idx:
            raise ValueError(
                "The class folders in backend/data/val must match "
                "backend/data/train."
            )
    else:
        evaluation_source = datasets.ImageFolder(
            train_dir, transform=EVALUATION_TRANSFORM
        )
        train_indices, validation_indices = stratified_split(
            train_dataset.targets, validation_fraction, seed
        )
        train_dataset = Subset(train_dataset, train_indices)
        validation_dataset = TransformedSubset(
            evaluation_source,
            validation_indices,
            EVALUATION_TRANSFORM,
        )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    return train_loader, validation_loader, class_to_idx


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("The validation dataset is empty.")
    return loss_sum / total, correct / total


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, validation_loader, class_to_idx = make_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.validation_fraction,
        args.seed,
        device,
    )
    model = build_model(len(class_to_idx), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        [
            {"params": model.fc.parameters(), "lr": HEAD_LEARNING_RATE},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("fc.")
                ],
                "lr": BASE_LEARNING_RATE,
            },
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[HEAD_LEARNING_RATE, BASE_LEARNING_RATE],
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_validation_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        approximate_correct = 0.0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="batch",
        )
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            permutation = torch.randperm(images.size(0), device=device)
            mixed_images = lam * images + (1.0 - lam) * images[permutation]
            labels_a, labels_b = labels, labels[permutation]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(mixed_images)
                loss = (
                    lam * criterion(logits, labels_a)
                    + (1.0 - lam) * criterion(logits, labels_b)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            approximate_correct += (
                lam * (logits.argmax(dim=1) == labels_a).sum().item()
                + (1.0 - lam) * (logits.argmax(dim=1) == labels_b).sum().item()
            )
            total += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
        )
        print(
            f"Epoch {epoch + 1}: "
            f"train loss={loss_sum / total:.4f}, "
            f"approx. train accuracy={approximate_correct / total:.2%}, "
            f"validation loss={validation_loss:.4f}, "
            f"validation accuracy={validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "epoch": epoch + 1,
                    "validation_accuracy": validation_accuracy,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ResNet50 car classifier using class folders."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--validation-fraction", type=float, default=VALIDATION_FRACTION
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize ResNet50 without downloading ImageNet weights.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be between 0 and 1")
    return args


if __name__ == "__main__":
    train(parse_args())
