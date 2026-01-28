# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ============================================
# 2. DEVICE (GPU IF AVAILABLE)
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================
# 3. DATA TRANSFORMS
# ============================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # lightweight for speed
    transforms.ToTensor()
])

# ============================================
# 4. LOAD CIFAR-100 DATASET
# ============================================
train_dataset = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,        # larger batch since GPU
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("CIFAR-100 dataset loaded")

# ============================================
# 5. LOAD PRETRAINED RESNET-18
# ============================================
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer
num_classes = 100
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)
print("ResNet-18 model ready")

# ============================================
# 6. LOSS FUNCTION AND OPTIMIZER
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

# ============================================
# 7. TRAINING LOOP (NORMAL TRAINING)
# ============================================
epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# ============================================
# 8. TESTING LOOP
# ============================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy on CIFAR-100: {accuracy:.2f}%")

# ============================================
# 9. PLOT TRAINING LOSS
# ============================================
plt.plot(train_losses, marker='o')
plt.title("Training Loss (ResNet-18 on CIFAR-100, GPU)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
