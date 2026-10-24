import torch
import torch.nn as nn
import timm

from bird_data import BirdDataset, label_to_idx
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import timm

from bird_data import BirdDataset, label_to_idx
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score
)

# Settings
BATCH_SIZE = 4
NUM_CLASSES = 182
EPOCHS = 3

DEVICE = torch.device("cpu")

# -------------------------
# Training dataset
# -------------------------
train_dataset = BirdDataset(
    "train_split.csv",
    "birdclef-2024/train_audio",
    label_to_idx
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# -------------------------
# Validation dataset
# -------------------------
val_dataset = BirdDataset(
    "val_split.csv",
    "birdclef-2024/train_audio",
    label_to_idx
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -------------------------
# Model
# -------------------------
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=NUM_CLASSES
)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Train classifier only
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

best_accuracy = 0

# -------------------------
# Training
# -------------------------
for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_number, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_number + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"Batch {batch_number + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = total_loss / len(train_loader)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total

    print()
    print(f"Epoch {epoch + 1} completed")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print()

    # Save best model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_baseline_model.pth"
        )

        print("Best model saved!")

print("Training completed.")
print("Best validation accuracy:", best_accuracy)


# -------------------------
# Final Test Set Evaluation
# -------------------------

model.eval()

y_true = []
y_pred = []
y_probs = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_probs.extend(probabilities.cpu().numpy())

# Calculate metrics

test_accuracy = accuracy_score(y_true, y_pred)

test_precision = precision_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

test_auc = roc_auc_score(
    y_true,
    y_probs,
    multi_class="ovr",
    average="macro"
)

print("\n===== FINAL TEST RESULTS =====")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")
print(f"ROC-AUC  : {test_auc:.4f}")