# ===============================
# TPU SETUP (KAGGLE TPU)
# ===============================
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# TPU Imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

device = xm.xla_device()
print("Using device:", device)

# ===============================
# DATASET
# ===============================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])

trainset = torchvision.datasets.CIFAR100(
    "./data", train=True, download=True, transform=train_transform
)

testset = torchvision.datasets.CIFAR100(
    "./data", train=False, download=True, transform=test_transform
)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# ===============================
# MODEL
# ===============================
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 100)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# ===============================
# ACCURACY FUNCTION
# ===============================
def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    para_loader = pl.ParallelLoader(loader, [device])
    loader_device = para_loader.per_device_loader(device)

    with torch.no_grad():
        for x, y in loader_device:
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100 * correct / total


# ===============================
# FREEZING FUNCTION
# ===============================
def freeze_until(layer_number):
    """
    layer_number:
    1 -> freeze layer1
    2 -> freeze layer1+layer2
    3 -> freeze layer1+2+3
    4 -> freeze layer1+2+3+4
    """

    # First unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    if layer_number >= 1:
        for param in model.layer1.parameters():
            param.requires_grad = False

    if layer_number >= 2:
        for param in model.layer2.parameters():
            param.requires_grad = False

    if layer_number >= 3:
        for param in model.layer3.parameters():
            param.requires_grad = False

    if layer_number >= 4:
        for param in model.layer4.parameters():
            param.requires_grad = False


# ===============================
# TRAINING FUNCTION
# ===============================
def train_epochs(epochs, freeze_level):

    freeze_until(freeze_level)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=1e-4
    )

    for epoch in range(epochs):

        model.train()
        loss_sum = 0.0

        para_loader = pl.ParallelLoader(trainloader, [device])
        loader_device = para_loader.per_device_loader(device)

        for x, y in loader_device:

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            xm.optimizer_step(optimizer)
            xm.mark_step()

            loss_sum += loss.item()

        train_acc = accuracy(model, trainloader)
        test_acc = accuracy(model, testloader)

        print(
            f"[Freeze Level {freeze_level}] "
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss {loss_sum/len(trainloader):.4f} | "
            f"Train {train_acc:.2f}% | "
            f"Test {test_acc:.2f}%"
        )


# ===============================
# INCREMENTAL FREEZING
# ===============================

print("\n🔥 Progressive Freezing Training")

# 0 = Full finetune
train_epochs(5, freeze_level=0)

# Freeze layer1
train_epochs(5, freeze_level=1)

# Freeze layer1 + layer2
train_epochs(5, freeze_level=2)

# Freeze layer1 + layer2 + layer3
train_epochs(5, freeze_level=3)

# Freeze layer1 + layer2 + layer3 + layer4
train_epochs(5, freeze_level=4)

print("✅ Training Complete")
