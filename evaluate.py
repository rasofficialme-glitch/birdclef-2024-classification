import torch
import timm
from torch.utils.data import DataLoader

from bird_data import BirdDataset, label_to_idx


# Settings
BATCH_SIZE = 4
NUM_CLASSES = 182
DEVICE = torch.device("cpu")

# Validation dataset
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

# Load model
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load("baseline_model.pth", map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

# Evaluate
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

accuracy = correct / total

print("Validation samples:", total)
print("Validation accuracy:", accuracy)