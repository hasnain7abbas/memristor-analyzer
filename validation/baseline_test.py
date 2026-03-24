"""
Phase 1A: Ideal PyTorch MLP baseline on real MNIST.

Architecture: 784 -> 256 -> 10 (configurable hidden size)
Activation:  ReLU on hidden layer, no activation on output (raw logits)
Loss:        CrossEntropyLoss (internally computes log-softmax)
Optimizer:   Adam, lr=0.001
Dataset:     torchvision MNIST, normalized to [0,1] via ToTensor
Batch size:  32
Epochs:      10

Pass/fail criteria:
  - Epoch 1:  > 90%
  - Epoch 3:  > 95%
  - Epoch 10: > 97%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------- hyperparameters ----------
HIDDEN_SIZE = 256
EPOCHS = 10
LR = 0.001
BATCH_SIZE = 32
SEED = 42

# ---------- reproducibility ----------
torch.manual_seed(SEED)

# ---------- data ----------
transform = transforms.ToTensor()  # scales [0,255] -> [0,1]

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28x28 -> 784
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits -- NO softmax
        return x


model = MLP(HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ---------- helpers ----------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100.0


# ---------- training ----------
results = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1

    acc = evaluate(model, test_loader)
    avg_loss = running_loss / num_batches
    results.append((epoch, acc, avg_loss))
    print(f"Epoch {epoch:2d} | Test Accuracy: {acc:.2f}% | Avg Loss: {avg_loss:.4f}")

# ---------- summary ----------
print("\n" + "=" * 50)
print("PASS/FAIL CHECK:")
print(f"  Epoch 1  accuracy: {results[0][1]:.2f}%  {'PASS' if results[0][1] > 90 else 'FAIL'} (need >90%)")
print(f"  Epoch 3  accuracy: {results[2][1]:.2f}%  {'PASS' if results[2][1] > 95 else 'FAIL'} (need >95%)")
print(f"  Epoch 10 accuracy: {results[9][1]:.2f}%  {'PASS' if results[9][1] > 97 else 'FAIL'} (need >97%)")
print("=" * 50)
