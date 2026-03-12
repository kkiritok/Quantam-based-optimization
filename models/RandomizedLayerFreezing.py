import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import random
import copy

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---------------- DATA TRANSFORMS ----------------
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.25)
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASETS ----------------
trainset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

testset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

# ---------------- MODEL ----------------
model = resnet18(weights="IMAGENET1K_V1")

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 100)
)

model = model.to(device)

# ---------------- FREEZE ALL ----------------
for p in model.parameters():
    p.requires_grad = False

# ---------------- ACCURACY FUNCTION ----------------
def accuracy_loader(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            _, pred = out.max(1)

            total += y.size(0)
            correct += pred.eq(y).sum().item()

    return correct / total


# ---------------- INITIAL ACCURACY ----------------
best_acc = accuracy_loader(model, testloader)
print("Initial Test Accuracy:", best_acc)

# ---------------- LAYERS ----------------
layers = {
    "layer1": model.layer1,
    "layer2": model.layer2,
    "layer3": model.layer3,
    "layer4": model.layer4
}

patience = 3
no_improve_count = 0
max_iterations = 40

# ---------------- TRAINING LOOP ----------------
for i in range(max_iterations):

    print("\n========== Iteration", i + 1, "==========")

    chosen_names = random.sample(
        list(layers.keys()),
        random.choice([1, 2])
    )

    print("Unfreezing:", chosen_names)

    old_weights = copy.deepcopy(model.state_dict())

    # -------- UNFREEZE RANDOM LAYERS --------
    for name in chosen_names:
        for p in layers[name].parameters():
            p.requires_grad = True

    # -------- ALWAYS TRAIN FC --------
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=8
    )

    # -------- LABEL SMOOTHING LOSS --------
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # -------- TRAIN --------
    model.train()

    for epoch in range(8):

        running_loss = 0

        for x, y in trainloader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = loss_fn(out, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1} Loss: {running_loss/len(trainloader):.4f}")

    # -------- EVALUATE --------
    train_acc = accuracy_loader(model, trainloader)
    test_acc = accuracy_loader(model, testloader)

    print("Train Accuracy:", train_acc)
    print("Test Accuracy :", test_acc)

    # -------- SAVE BEST MODEL --------
    if test_acc > best_acc:

        best_acc = test_acc
        no_improve_count = 0

        torch.save(model.state_dict(), "best_model.pth")

        print("✅ Improved — Model Saved")

    else:

        model.load_state_dict(old_weights)

        no_improve_count += 1

        print("❌ Reverted | No Improve Count:", no_improve_count)

    # -------- FREEZE AGAIN --------
    for p in model.parameters():
        p.requires_grad = False

    # -------- EARLY STOPPING --------
    if no_improve_count >= patience:

        print("\n🛑 Early stopping triggered!")

        break


print("\nFinal Best Accuracy:", best_acc)
print("Best model saved as best_model.pth")
