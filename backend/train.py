import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ----------------------
# Device Setup
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA GPU not available. Using CPU, training will be slower.")

# ----------------------
# Paths
# ----------------------
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = Path("model.pth")

# ----------------------
# Settings
# ----------------------
BATCH_SIZE = 64   # Larger batch since you have strong GPU
NUM_EPOCHS = 40
BASE_LR = 1e-4
MAX_LR = 3e-4  # for OneCycle
MIXUP_ALPHA = 0.4  # MixUp strength

# ----------------------
# Data Augmentation
# ----------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------------------
# Dataset & Loader
# ----------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

# ----------------------
# Model (ResNet50)
# ----------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace final FC layer
num_classes = len(train_dataset.classes)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

# ----------------------
# Loss + Optimizer + Scheduler
# ----------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Discriminative learning rate: lower for backbone, higher for head
optimizer = optim.AdamW([
    {"params": model.fc.parameters(), "lr": MAX_LR},
    {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": BASE_LR}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LR,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS
)

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

# ----------------------
# MixUp function
# ----------------------
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ----------------------
# Training
# ----------------------
def train():
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # MixUp
            images, targets_a, targets_b, lam = mixup_data(images, labels)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (lam * preds.eq(targets_a).sum().item()
                        + (1 - lam) * preds.eq(targets_b).sum().item())
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Train Acc (approx): {acc:.2f}%")

        test_acc = evaluate()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx
            }, MODEL_PATH)
            print(f"✅ Saved new best model: {best_acc:.2f}%")

# ----------------------
# Evaluation
# ----------------------
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"🔎 Test Accuracy: {acc:.2f}%")
    return acc

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    train()
    print("Training complete.")
    evaluate()
    print("Final evaluation complete.")
