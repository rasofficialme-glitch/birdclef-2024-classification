import torch
import timm

from bird_data import BirdDataset, label_to_idx
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Settings
BATCH_SIZE = 4
NUM_CLASSES = 182

DEVICE = torch.device("cpu")

# -------------------------
# Test dataset
# -------------------------
test_dataset = BirdDataset(
    "val_split.csv",
    "birdclef-2024/train_audio",
    label_to_idx
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -------------------------
# Load model
# -------------------------
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        "best_baseline_model.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

# -------------------------
# Evaluation
# -------------------------
y_true = []
y_pred = []
y_probs = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_probs.extend(probabilities.cpu().numpy())

# -------------------------
# Calculate metrics
# -------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_probs,
    multi_class="ovr",
    average="macro"
)

# -------------------------
# Print results
# -------------------------

print("\n===== TEST SET RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")