import timm
import torch

NUM_CLASSES = 182

# Load EfficientNet-B0
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=NUM_CLASSES
)

print(model)

# Test with a dummy input
x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)