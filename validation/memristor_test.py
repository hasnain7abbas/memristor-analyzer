"""
Phase 1B: Memristor-aware PyTorch MLP on real MNIST.

Same architecture and training as baseline_test.py, but after every
optimizer.step() the weights are mapped through the memristor device model:

  1. Normalize weights to [0,1] based on current min/max
  2. Map to conductance range [G_min, G_max]
  3. Quantize to N_P * N_D uniform levels
  4. Add multiplicative cycle-to-cycle noise: G * (1 + sigma_w * randn)
  5. Clamp to [G_min, G_max]
  6. Map back to original weight space

Default BFO device parameters:
  G_min   = 0.1306e-6 S
  G_max   = 0.2557e-6 S
  sigma_w = 0.047213
  N_P     = 22
  N_D     = 6

Expected:
  - Ideal curve:    > 97% by epoch 10
  - Memristor curve: 40-90%, consistently lower than ideal every epoch
  - Reproducible within +/-3% across runs
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

# ---------- BFO device defaults ----------
G_MIN = 0.1306e-6
G_MAX = 0.2557e-6
SIGMA_W = 0.047213
N_P = 22
N_D = 6

# ---------- reproducibility ----------
torch.manual_seed(SEED)

# ---------- data ----------
transform = transforms.ToTensor()

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
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits
        return x


# ---------- device mapping ----------
def apply_device_mapping(model, g_min=G_MIN, g_max=G_MAX, sigma_w=SIGMA_W,
                         n_p=N_P, n_d=N_D):
    """Apply memristor device mapping to all weight tensors (not biases)."""
    num_levels = n_p * n_d

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue

            W = param.data
            w_min = W.min()
            w_max = W.max()

            if (w_max - w_min).abs() < 1e-15:
                continue

            # 1. Normalize to [0, 1]
            W_norm = (W - w_min) / (w_max - w_min)

            # 2. Quantize to discrete conductance levels
            G_index = torch.round(W_norm * (num_levels - 1))
            G_quantized = g_min + (G_index / (num_levels - 1)) * (g_max - g_min)

            # 3. Add multiplicative cycle-to-cycle noise
            noise = torch.randn_like(G_quantized) * sigma_w
            G_noisy = G_quantized * (1.0 + noise)
            G_clamped = G_noisy.clamp(g_min, g_max)

            # 4. Map back to weight space
            W_new = w_min + (w_max - w_min) * (G_clamped - g_min) / (g_max - g_min)
            param.data.copy_(W_new)


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


# ---------- create two networks from same init ----------
ideal_model = MLP(HIDDEN_SIZE)
memristor_model = MLP(HIDDEN_SIZE)

# Copy ideal weights to memristor model so they start identical
memristor_model.load_state_dict(ideal_model.state_dict())

criterion = nn.CrossEntropyLoss()
ideal_optimizer = optim.Adam(ideal_model.parameters(), lr=LR)
mem_optimizer = optim.Adam(memristor_model.parameters(), lr=LR)

# ---------- training ----------
ideal_results = []
mem_results = []

for epoch in range(1, EPOCHS + 1):
    ideal_model.train()
    memristor_model.train()

    ideal_loss_sum = 0.0
    mem_loss_sum = 0.0
    num_batches = 0

    for images, labels in train_loader:
        # --- Ideal network ---
        ideal_optimizer.zero_grad()
        ideal_logits = ideal_model(images)
        ideal_loss = criterion(ideal_logits, labels)
        ideal_loss.backward()
        ideal_optimizer.step()
        ideal_loss_sum += ideal_loss.item()

        # --- Memristor network ---
        mem_optimizer.zero_grad()
        mem_logits = memristor_model(images)
        mem_loss = criterion(mem_logits, labels)
        mem_loss.backward()
        mem_optimizer.step()

        # Apply device mapping AFTER optimizer step
        apply_device_mapping(memristor_model)

        mem_loss_sum += mem_loss.item()
        num_batches += 1

    ideal_acc = evaluate(ideal_model, test_loader)
    mem_acc = evaluate(memristor_model, test_loader)

    ideal_results.append((epoch, ideal_acc, ideal_loss_sum / num_batches))
    mem_results.append((epoch, mem_acc, mem_loss_sum / num_batches))

    print(f"Epoch {epoch:2d} | Ideal: {ideal_acc:.2f}% | Memristor: {mem_acc:.2f}%")

# ---------- comparison table ----------
print("\n" + "=" * 50)
print(f"{'Epoch':<8}| {'Ideal (%)':>10} | {'Memristor (%)':>14}")
print("-" * 8 + "+" + "-" * 12 + "+" + "-" * 16)
for (e, ia, _), (_, ma, _) in zip(ideal_results, mem_results):
    print(f"{e:<8}| {ia:>10.1f} | {ma:>14.1f}")
print("=" * 50)

# ---------- validation ----------
print("\nVALIDATION:")
ideal_pass = ideal_results[9][1] > 97
mem_lower = all(m[1] <= i[1] for i, m in zip(ideal_results, mem_results))
mem_not_random = mem_results[9][1] > 20

print(f"  Ideal > 97% at epoch 10:         {ideal_results[9][1]:.2f}%  {'PASS' if ideal_pass else 'FAIL'}")
print(f"  Memristor always < ideal:         {'PASS' if mem_lower else 'FAIL'}")
print(f"  Memristor not stuck at random:    {mem_results[9][1]:.2f}%  {'PASS' if mem_not_random else 'FAIL'}")
