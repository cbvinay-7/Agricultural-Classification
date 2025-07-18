import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------- CONFIG ---------------- #
DATA_DIR = "dataset/train"  # 📁 Should have subfolders for each class
VAL_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORMS ---------------- #
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- LOAD DATA ---------------- #
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
NUM_CLASSES = len(full_dataset.classes)
print(f"📊 Detected {NUM_CLASSES} classes: {full_dataset.classes}")

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ---------------- MODEL SETUP ---------------- #
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ---------------- OPTIMIZER & LOSS ---------------- #
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2)

# ---------------- TRAIN LOOP ---------------- #
best_val_acc = 0.0
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # SAFETY CHECK
        if labels.max() >= NUM_CLASSES:
            print(f"⚠️ Invalid label detected: {labels}")
            continue

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---------------- VALIDATION ---------------- #
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    print(f"📅 Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | ✅ Val Acc: {val_acc:.4f}")

    # ---------------- SAVE BEST MODEL ---------------- #
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "convnext_agriculture_model.pt")
        print("💾 Best model saved!")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

print(f"✅ Training Complete. 🏁 Best Validation Accuracy: {best_val_acc:.4f}")
