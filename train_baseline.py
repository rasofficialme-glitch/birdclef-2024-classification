import torch
import torch.nn as nn
import timm

from bird_data import BirdDataset, label_to_idx
from torch.utils.data import DataLoader


# -------------------------
# Settings
# -------------------------
BATCH_SIZE = 4
NUM_CLASSES = 182
NUM_BATCHES = 20

DEVICE = torch.device("cpu")

# -------------------------
# Dataset
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
# Model
# -------------------------
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=NUM_CLASSES
)

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# Train only classifier
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# -------------------------
# Loss and optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# -------------------------
# Training
# -------------------------
model.train()

total_loss = 0

for batch_number, (images, labels) in enumerate(train_loader):

    if batch_number >= NUM_BATCHES:
        break

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    total_loss += loss.item()

    print(
        f"Batch {batch_number + 1}/{NUM_BATCHES} "
        f"Loss: {loss.item():.4f}"
    )

print("\nTraining test completed!")
print("Average loss:", total_loss / NUM_BATCHES)

# Save model
torch.save(
    model.state_dict(),
    "baseline_model.pth"
)

print("Model saved as baseline_model.pth")