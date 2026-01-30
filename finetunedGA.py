import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
import random

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Dataset
# ======================
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])

trainset = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset, batch_size=64, shuffle=False)

# ======================
# ResNet-18 Feature Extractor
# ======================
model = resnet18(pretrained=True)
model.fc = nn.Identity()
model = model.to(device)
model.eval()

for p in model.parameters():
    p.requires_grad = False

# ======================
# Feature Extraction
# ======================
def extract_features(loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = model(x)
            feats.append(f.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

print("Extracting features...")
X_train, y_train = extract_features(trainloader)
X_test,  y_test  = extract_features(testloader)

# ======================
# Train / Val Split (IMPORTANT)
# ======================
N = int(0.9 * len(X_train))
X_tr, X_val = X_train[:N], X_train[N:]
y_tr, y_val = y_train[:N], y_train[N:]

# ======================
# STEP 1: SGD Classifier (GPU)
# ======================
classifier = nn.Linear(512, 100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

loader = DataLoader(
    TensorDataset(X_tr.to(device), y_tr.to(device)),
    batch_size=256, shuffle=True
)

print("\n=== SGD TRAINING ===")
for epoch in range(20):
    loss_sum = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(classifier(xb), yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch+1}, Loss: {loss_sum/len(loader):.4f}")

# Save SGD weights
W_sgd = classifier.weight.data.T.cpu()   # (512 × 100)
b_sgd = classifier.bias.data.cpu()

# ======================
# STEP 2: ORTHOGONAL RANDOM PROJECTION
# ======================
D_HIGH = 2000

R = torch.randn(512, D_HIGH)
R, _ = torch.linalg.qr(R)   # 🔥 orthogonal

X_tr_h  = X_tr @ R
X_val_h = X_val @ R
X_te_h  = X_test @ R

W_high_init = R.T @ W_sgd   # (D_HIGH × 100)

# ======================
# STEP 3: GENETIC ALGORITHM (VALIDATION-BASED)
# ======================
POP_SIZE = 20
GENERATIONS = 20
MUT_RATE = 0.3
DELTA = 0.01

def fitness(W):
    with torch.no_grad():
        logits = X_val_h @ W
        loss = criterion(logits, y_val)
    return -loss.item()

population = [
    W_high_init + 0.01 * torch.randn_like(W_high_init)
    for _ in range(POP_SIZE)
]

print("\n=== GA OPTIMIZATION ===")
for gen in range(GENERATIONS):
    scored = sorted([(fitness(W), W) for W in population], key=lambda x: x[0], reverse=True)
    parents = [W for _, W in scored[:5]]
    new_pop = parents.copy()

    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(parents, 2)
        child = 0.5 * (p1 + p2)

        if random.random() < MUT_RATE:
            child += DELTA * torch.randn_like(child)

        new_pop.append(child)

    population = new_pop
    print(f"GA Gen {gen+1}, Best Val Loss: {-scored[0][0]:.4f}")

best_W_high = scored[0][1]

# ======================
# STEP 4: PROJECT BACK
# ======================
W_final = R @ best_W_high   # (512 × 100)

# ======================
# STEP 5: FINAL SGD FINE-TUNING (🔥 CRITICAL)
# ======================
classifier = nn.Linear(512, 100).to(device)
classifier.weight.data = W_final.T.to(device)
classifier.bias.data = b_sgd.to(device)

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

loader = DataLoader(
    TensorDataset(X_tr.to(device), y_tr.to(device)),
    batch_size=256, shuffle=True
)

print("\n=== FINAL SGD FINETUNE ===")
for epoch in range(10):
    loss_sum = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(classifier(xb), yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Finetune Epoch {epoch+1}, Loss: {loss_sum/len(loader):.4f}")

# ======================
# FINAL EVALUATION
# ======================
classifier.eval()
with torch.no_grad():
    logits = classifier(X_test.to(device))
    acc = (logits.argmax(1).cpu() == y_test).float().mean()

print("\n🔥 FINAL TEST ACCURACY:", acc.item() * 100)
