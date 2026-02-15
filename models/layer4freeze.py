import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Dataset
# ======================
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

# ======================
# Model (FREEZE ONLY layer4)
# ======================
model = resnet18(pretrained=True)

# Freeze ONLY layer4
for param in model.layer4.parameters():
    param.requires_grad = False

# Replace classifier for CIFAR-100
model.fc = nn.Linear(512, 100)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Optimizer will update only trainable params
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

# ======================
# Accuracy function
# ======================
def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# ======================
# Training Loop
# ======================
EPOCHS = 10

print("\n🚀 FINETUNING (layer4 FROZEN)")
for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0.0

    for x, y in trainloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    train_acc = accuracy(model, trainloader)
    test_acc = accuracy(model, testloader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss {loss_sum/len(trainloader):.4f} | "
        f"Train Acc {train_acc:.2f}% | "
        f"Test Acc {test_acc:.2f}%"
    )
