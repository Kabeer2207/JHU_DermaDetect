import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

MODEL_VERSION = "v3"   #ringworm and vitiligo added


# =========================
# Configuration (NO hardcoding)
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent        # backend/
PROJECT_ROOT = BASE_DIR.parent                   # JHU_project/

DATASET_DIR = PROJECT_ROOT / "data" / "final_split"
MODEL_DIR = PROJECT_ROOT / "models"

for split in ["train", "val", "test"]:
    p = DATASET_DIR / split
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset split: {p}")


BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-5
MODEL_NAME = "resnet50"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# Transforms
# =========================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Dataset loading (AUTO class discovery)
# =========================

train_dir = Path(DATASET_DIR) / "train"
val_dir   = Path(DATASET_DIR) / "val"
test_dir  = Path(DATASET_DIR) / "test"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset  = datasets.ImageFolder(test_dir, transform=val_test_transforms)

class_names = train_dataset.classes
num_classes = len(class_names)

print("✅ Classes discovered:")
for i, cls in enumerate(class_names):
    print(f"  {i}: {cls}")

# Save class mapping for inference
class_map_path = Path(MODEL_DIR) / "class_names.json"
with open(class_map_path, "w") as f:
    json.dump(class_names, f, indent=2)

print(f"\n📁 Class mapping saved to: {class_map_path}")

# =========================
# Dataloaders
# =========================

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# Model
# =========================

def get_model(num_classes: int):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True


    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

model = get_model(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [
        {"params": model.layer4.parameters(), "lr": LEARNING_RATE},
        {"params": model.fc.parameters(), "lr": LEARNING_RATE},
    ]
)


# =========================
# Training loop
# =========================

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\n🟦 Epoch {epoch + 1}/{EPOCHS}")

    # ---- Training ----
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_train_loss = train_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)

    print(
        f"📊 Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # ---- Save best model ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_path = Path(MODEL_DIR) / f"resnet50_{MODEL_VERSION}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved best model → {model_path}")

# =========================
# Test evaluation
# =========================

model.load_state_dict(
    torch.load(Path(MODEL_DIR) / f"resnet50_{MODEL_VERSION}.pth")
)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")

print("\n🎉 Training complete.")
print(f"📦 Model saved in: {MODEL_DIR}")
print(f"📄 Classes used: {class_names}")
