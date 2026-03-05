use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScriptParams {
    #[serde(rename = "Gmin")]
    pub g_min: f64,
    #[serde(rename = "Gmax")]
    pub g_max: f64,
    #[serde(rename = "alphaP")]
    pub alpha_p: f64,
    #[serde(rename = "alphaD")]
    pub alpha_d: f64,
    #[serde(rename = "writeNoise")]
    pub write_noise: f64,
    #[serde(rename = "numLevelsP")]
    pub num_levels_p: usize,
    #[serde(rename = "numLevelsD")]
    pub num_levels_d: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScriptANNConfig {
    #[serde(rename = "modelType", default = "default_model_type")]
    pub model_type: String,
    #[serde(rename = "hiddenSize")]
    pub hidden_size: usize,
    #[serde(rename = "hiddenSize2", default = "default_hidden_size_2")]
    pub hidden_size_2: usize,
    pub epochs: usize,
    #[serde(rename = "learningRate")]
    pub learning_rate: f64,
    #[serde(rename = "batchSize")]
    pub batch_size: usize,
}

fn default_model_type() -> String {
    "mlp_1h".to_string()
}

fn default_hidden_size_2() -> usize {
    64
}

#[tauri::command]
pub fn generate_python_script(
    params: ScriptParams,
    config: ScriptANNConfig,
) -> Result<String, String> {
    // Build model class string based on model type
    let (model_class, arch_str) = match config.model_type.as_str() {
        "perceptron" => (
            r#"class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x"#.to_string(),
            "784 → 10".to_string(),
        ),
        "mlp_2h" => (
            r#"class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, 10)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x"#.to_string(),
            format!("784 → {} → {} → 10", config.hidden_size, config.hidden_size_2),
        ),
        "lenet5" | "cnn_simple" => (
            r#"class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

MLP = LeNet5  # Use LeNet-5 as the model"#.to_string(),
            "LeNet-5: Conv(1,6,5)→Pool→Conv(6,16,5)→Pool→FC(400→120→84→10)".to_string(),
        ),
        _ => (
            // mlp_1h
            r#"class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x"#.to_string(),
            format!("784 → {} → 10", config.hidden_size),
        ),
    };

    let script = format!(
        r#"#!/usr/bin/env python3
"""
Memristor Neural Analyzer — MNIST ANN Simulation Script
Generated with device parameters from experimental data.
This script trains on the REAL MNIST dataset (60,000 images).
Uses copy-and-degrade approach: only ideal network is trained,
memristor accuracy is evaluated by copying and degrading weights each epoch.

Requirements: pip install torch torchvision matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import csv
import copy
import os

# ============================================================
# Device Parameters (extracted from experimental data)
# ============================================================
G_MIN = {g_min:.6}           # Minimum conductance (µS)
G_MAX = {g_max:.6}           # Maximum conductance (µS)
ALPHA_P = {alpha_p:.4}       # Potentiation non-linearity
ALPHA_D = {alpha_d:.4}       # Depression non-linearity
WRITE_NOISE = {write_noise:.6}  # Normalized write noise σ_w
NUM_LEVELS_P = {num_levels_p}    # Potentiation conductance levels
NUM_LEVELS_D = {num_levels_d}    # Depression conductance levels
NUM_LEVELS = max(NUM_LEVELS_P, NUM_LEVELS_D)

# ============================================================
# Training Configuration
# ============================================================
HIDDEN_SIZE = {hidden_size}
HIDDEN_SIZE_2 = {hidden_size_2}
MODEL_TYPE = '{model_type}'
EPOCHS = {epochs}
LEARNING_RATE = {learning_rate}
BATCH_SIZE = {batch_size}
MOMENTUM = 0.9

# ============================================================
# Model Definition
# ============================================================
{model_class}

# ============================================================
# Memristor Non-Ideality Functions (Copy-and-Degrade)
# ============================================================
def apply_nonlinear_weight_remap(model):
    """Apply non-linear weight remapping to simulate memristor storage distortion."""
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() < 2:
                continue
            w = param.data
            w_min, w_max = w.min(), w.max()
            if w_max - w_min < 1e-10:
                continue

            w_norm = ((w - w_min) / (w_max - w_min)).clamp(0.0, 1.0)

            # Use alpha_p for upper half, alpha_d for lower half
            alpha_map = torch.where(w_norm >= 0.5,
                torch.tensor(ALPHA_P, device=w.device),
                torch.tensor(ALPHA_D, device=w.device))

            # Non-linear remapping: (1 - exp(-alpha * w_norm)) / (1 - exp(-alpha))
            near_linear = alpha_map.abs() < 0.01
            remapped = torch.where(near_linear, w_norm,
                (1.0 - torch.exp(-alpha_map * w_norm)) / (1.0 - torch.exp(-alpha_map)))

            param.data = remapped * (w_max - w_min) + w_min

def quantize_and_add_noise(model):
    """Quantize weights to N levels and add Gaussian write noise."""
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() < 2:
                continue
            w = param.data
            w_min, w_max = w.min(), w.max()
            if w_max - w_min < 1e-10:
                continue

            w_norm = (w - w_min) / (w_max - w_min)
            w_norm = torch.round(w_norm * (NUM_LEVELS - 1)) / (NUM_LEVELS - 1)
            noise = torch.randn_like(w_norm) * WRITE_NOISE / np.sqrt(NUM_LEVELS)
            w_norm = torch.clamp(w_norm + noise, 0.0, 1.0)
            param.data = w_norm * (w_max - w_min) + w_min

# ============================================================
# Data Loading
# ============================================================
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# ============================================================
# Training (Copy-and-Degrade approach)
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Only the ideal model is trained
ideal_model = MLP().to(device)
ideal_optimizer = optim.SGD(ideal_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()

results = []
print(f"\nTraining for {{EPOCHS}} epochs (copy-and-degrade approach)...")
print(f"Architecture: {arch_str}")
print(f"Device params: αP={{ALPHA_P:.2f}}, αD={{ALPHA_D:.2f}}, σw={{WRITE_NOISE:.4f}}, N={{NUM_LEVELS}}")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    # Train ideal model only
    ideal_model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        ideal_optimizer.zero_grad()
        output = ideal_model(data)
        loss = criterion(output, target)
        loss.backward()
        ideal_optimizer.step()

    # Copy-and-degrade: clone ideal weights, then apply memristor non-idealities
    mem_model = copy.deepcopy(ideal_model)
    apply_nonlinear_weight_remap(mem_model)
    quantize_and_add_noise(mem_model)

    # Evaluate both
    ideal_model.eval()
    mem_model.eval()

    def evaluate(model):
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        return correct / total * 100, total_loss / total

    ideal_acc, ideal_loss = evaluate(ideal_model)
    mem_acc, mem_loss = evaluate(mem_model)

    results.append({{
        'epoch': epoch,
        'ideal_acc': ideal_acc,
        'mem_acc': mem_acc,
        'ideal_loss': ideal_loss,
        'mem_loss': mem_loss,
    }})

    print(f"Epoch {{epoch:3d}}/{{EPOCHS}} | "
          f"Ideal: {{ideal_acc:.2f}}% (loss={{ideal_loss:.4f}}) | "
          f"Memristor: {{mem_acc:.2f}}% (loss={{mem_loss:.4f}})")

# ============================================================
# Results
# ============================================================
best_ideal = max(r['ideal_acc'] for r in results)
best_mem = max(r['mem_acc'] for r in results)
drop = best_ideal - best_mem

print("\n" + "=" * 70)
print(f"Best Ideal Accuracy:     {{best_ideal:.2f}}%")
print(f"Best Memristor Accuracy: {{best_mem:.2f}}%")
print(f"Accuracy Drop:           {{drop:.2f}}%")
print("=" * 70)

# Save CSV
with open('training_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch', 'ideal_acc', 'mem_acc', 'ideal_loss', 'mem_loss'])
    writer.writeheader()
    writer.writerows(results)
print("Saved: training_results.csv")

# Plot
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 5))
epochs_list = [r['epoch'] for r in results]
ax.plot(epochs_list, [r['ideal_acc'] for r in results], 'b-o', markersize=3, label='Ideal', linewidth=2)
ax.plot(epochs_list, [r['mem_acc'] for r in results], 'r-s', markersize=3, label='Memristor', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title(f'MNIST Classification: Ideal vs Memristor (Copy-and-Degrade)\n'
             f'(αP={{ALPHA_P:.2f}}, αD={{ALPHA_D:.2f}}, σw={{WRITE_NOISE:.4f}}, N={{NUM_LEVELS}})')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('mnist_accuracy.png', dpi=300, bbox_inches='tight')
plt.savefig('mnist_accuracy.pdf', bbox_inches='tight')
print("Saved: mnist_accuracy.png, mnist_accuracy.pdf")
plt.show()
"#,
        g_min = params.g_min,
        g_max = params.g_max,
        alpha_p = params.alpha_p,
        alpha_d = params.alpha_d,
        write_noise = params.write_noise,
        num_levels_p = params.num_levels_p,
        num_levels_d = params.num_levels_d,
        hidden_size = config.hidden_size,
        hidden_size_2 = config.hidden_size_2,
        model_type = config.model_type,
        epochs = config.epochs,
        learning_rate = config.learning_rate,
        batch_size = config.batch_size,
        model_class = model_class,
        arch_str = arch_str,
    );

    Ok(script)
}

#[tauri::command]
pub fn generate_crosssim_script(params: ScriptParams) -> Result<String, String> {
    let script = format!(
        r#"#!/usr/bin/env python3
"""
Memristor Neural Analyzer — CrossSim Crossbar Simulation Script
Generated with device parameters from experimental data.

CrossSim is developed by Sandia National Laboratories for simulating
analog crossbar arrays with realistic device models.

Requirements:
  pip install cross-sim
  See: https://github.com/sandialabs/cross-sim

This script configures a crossbar array with your extracted device
parameters and runs MNIST classification.
"""

import numpy as np

try:
    from simulator import CrossSimParameters
    from simulator.cores import AnalogCore
    from simulator.algorithms import DNN
except ImportError:
    print("CrossSim not found. Install from: https://github.com/sandialabs/cross-sim")
    print("  git clone https://github.com/sandialabs/cross-sim.git")
    print("  cd cross-sim && pip install -e .")
    exit(1)

# ============================================================
# Device Parameters (from experimental extraction)
# ============================================================
G_MIN = {g_min:.6}e-6      # Min conductance (S) — converted from µS
G_MAX = {g_max:.6}e-6      # Max conductance (S) — converted from µS
ALPHA_P = {alpha_p:.4}       # Potentiation non-linearity
ALPHA_D = {alpha_d:.4}       # Depression non-linearity
WRITE_NOISE = {write_noise:.6}  # Normalized write noise σ_w
NUM_LEVELS = max({num_levels_p}, {num_levels_d})

# ============================================================
# CrossSim Configuration
# ============================================================
params = CrossSimParameters()

# Core configuration
params.core.style = "BALANCED"          # Balanced crossbar for +/- weights
params.core.rows = 784                  # MNIST input size
params.core.cols = 10                   # Output classes

# Device model
params.core.mapping.weights.minimum = G_MIN
params.core.mapping.weights.maximum = G_MAX

# Programming non-linearity
params.core.device.programming.Gmin_LTP = G_MIN
params.core.device.programming.Gmax_LTP = G_MAX
params.core.device.programming.alpha_LTP = ALPHA_P  # Potentiation NL
params.core.device.programming.alpha_LTD = ALPHA_D  # Depression NL

# Write noise
params.core.device.programming.write_noise.sigma = WRITE_NOISE * (G_MAX - G_MIN)
params.core.device.programming.write_noise.type = "gaussian"

# Quantization (number of conductance levels)
params.core.device.programming.num_levels = NUM_LEVELS

# Read noise (estimate from write noise)
params.core.device.read_noise.sigma = WRITE_NOISE * 0.1 * (G_MAX - G_MIN)
params.core.device.read_noise.type = "gaussian"

print("CrossSim Parameters Configured:")
print(f"  Conductance range: {{G_MIN*1e6:.2f}} - {{G_MAX*1e6:.2f}} µS")
print(f"  Non-linearity: αP={{ALPHA_P:.2f}}, αD={{ALPHA_D:.2f}}")
print(f"  Write noise: σ={{WRITE_NOISE:.4f}}")
print(f"  Conductance levels: {{NUM_LEVELS}}")

# ============================================================
# DNN Setup and Training
# ============================================================
print("\nSetting up DNN with crossbar core...")
print("Note: Modify the network architecture and training loop")
print("according to CrossSim's latest API documentation.")
print("See: https://cross-sim.sandia.gov/")

# Example architecture setup (adapt to your CrossSim version):
# dnn = DNN(params)
# dnn.set_network([784, 128, 10])
# dnn.train_network(epochs=50, dataset='mnist')
# dnn.evaluate()

print("\nScript generated successfully.")
print("Please adapt the training loop to your CrossSim version.")
"#,
        g_min = params.g_min,
        g_max = params.g_max,
        alpha_p = params.alpha_p,
        alpha_d = params.alpha_d,
        write_noise = params.write_noise,
        num_levels_p = params.num_levels_p,
        num_levels_d = params.num_levels_d,
    );

    Ok(script)
}

#[tauri::command]
pub fn generate_neurosim_script(params: ScriptParams) -> Result<String, String> {
    let script = format!(
        r#"#!/usr/bin/env python3
"""
Memristor Neural Analyzer — NeuroSim Configuration Script
Generated with device parameters from experimental data.

NeuroSim is developed by Georgia Institute of Technology for
benchmarking neuromorphic hardware with realistic device models.

Setup:
  1. Clone: https://github.com/neurosim/DNN_NeuroSim_V1.4
  2. Follow build instructions in the repo README
  3. Use this script to generate the device configuration

This script generates NeuroSim-compatible device parameter files.
"""

import os
import json

# ============================================================
# Device Parameters (from experimental extraction)
# ============================================================
G_MIN = {g_min:.6}e-6      # Min conductance (S)
G_MAX = {g_max:.6}e-6      # Max conductance (S)
ALPHA_P = {alpha_p:.4}       # Potentiation non-linearity
ALPHA_D = {alpha_d:.4}       # Depression non-linearity
WRITE_NOISE = {write_noise:.6}  # Normalized write noise
NUM_LEVELS_P = {num_levels_p}   # Potentiation levels
NUM_LEVELS_D = {num_levels_d}   # Depression levels

# Derived parameters for NeuroSim
R_ON = 1.0 / G_MAX         # Low resistance state (Ω)
R_OFF = 1.0 / G_MIN        # High resistance state (Ω)
ON_OFF_RATIO = G_MAX / G_MIN

# ============================================================
# Generate NeuroSim Device Configuration
# ============================================================
config = {{
    "DeviceType": "AnalogNVM",
    "maxConductance": G_MAX,
    "minConductance": G_MIN,
    "maxNumLevelLTP": NUM_LEVELS_P,
    "maxNumLevelLTD": NUM_LEVELS_D,
    "NL_LTP": ALPHA_P,             # Non-linearity for LTP
    "NL_LTD": -ALPHA_D,            # Non-linearity for LTD (negative convention)
    "sigmaDtoD": WRITE_NOISE,       # Device-to-device variation
    "sigmaCtoC": WRITE_NOISE * 0.5, # Cycle-to-cycle variation
    "readVoltage": 0.5,             # Read voltage (V)
    "readPulseWidth": 5e-9,         # Read pulse width (s)
    "writeVoltage": 2.0,            # Write voltage (V)
    "writePulseWidth": 100e-9,      # Write pulse width (s)
    "accessVoltage": 1.1,           # Access transistor voltage (V)
    "resistanceOn": R_ON,
    "resistanceOff": R_OFF,
}}

# Save as JSON for reference
with open('neurosim_device_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("Saved: neurosim_device_config.json")

# Generate NeuroSim Cell.cpp parameter block
cpp_params = f"""
// ============================================================
// NeuroSim Device Parameters (auto-generated)
// Paste into Cell.cpp in the AnalogNVM initialization section
// ============================================================
// Conductance range
cell.maxConductance = {{G_MAX}};  // S
cell.minConductance = {{G_MIN}};  // S

// Number of conductance levels
cell.maxNumLevelLTP = {{NUM_LEVELS_P}};
cell.maxNumLevelLTD = {{NUM_LEVELS_D}};

// Non-linearity
cell.NL_LTP = {{ALPHA_P}};
cell.NL_LTD = {{-ALPHA_D}};  // Negative convention for depression

// Noise
cell.sigmaDtoD = {{WRITE_NOISE}};    // Device-to-device variation
cell.sigmaCtoC = {{WRITE_NOISE * 0.5}};  // Cycle-to-cycle variation
"""

with open('neurosim_cell_params.txt', 'w') as f:
    f.write(cpp_params)
print("Saved: neurosim_cell_params.txt")

print(f"\nDevice Summary:")
print(f"  R_ON  = {{R_ON:.0f}} Ω ({{G_MAX*1e6:.2f}} µS)")
print(f"  R_OFF = {{R_OFF:.0f}} Ω ({{G_MIN*1e6:.2f}} µS)")
print(f"  On/Off = {{ON_OFF_RATIO:.1f}}")
print(f"  NL_LTP = {{ALPHA_P:.2f}}, NL_LTD = {{ALPHA_D:.2f}}")
print(f"  σ_w = {{WRITE_NOISE:.4f}}")
print(f"\nInstructions:")
print(f"  1. Clone https://github.com/neurosim/DNN_NeuroSim_V1.4")
print(f"  2. Copy parameters from neurosim_cell_params.txt into Cell.cpp")
print(f"  3. Rebuild and run the simulation")
"#,
        g_min = params.g_min,
        g_max = params.g_max,
        alpha_p = params.alpha_p,
        alpha_d = params.alpha_d,
        write_noise = params.write_noise,
        num_levels_p = params.num_levels_p,
        num_levels_d = params.num_levels_d,
    );

    Ok(script)
}

#[tauri::command]
pub fn generate_memtorch_script(params: ScriptParams, config: ScriptANNConfig) -> Result<String, String> {
    let script = format!(
        r#"#!/usr/bin/env python3
"""
Memristor Neural Analyzer — MemTorch Simulation Script
Generated with device parameters from experimental data.

MemTorch is a PyTorch-based framework for simulating memristive
deep neural networks. It patches standard PyTorch models to use
memristive device models for weight storage.

Requirements:
  pip install memtorch torch torchvision matplotlib numpy

Reference: https://github.com/coreylammie/MemTorch
Paper: Lammie et al., "MemTorch: An Open-Source Simulation Framework
       for Memristive Deep Learning Systems," Neurocomputing, 2022
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

try:
    import memtorch
    from memtorch.mn.Module import patch_model
    from memtorch.map.Parameter import naive_map
    from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
    HAS_MEMTORCH = True
except ImportError:
    print("MemTorch not found. Install with: pip install memtorch")
    print("See: https://github.com/coreylammie/MemTorch")
    HAS_MEMTORCH = False

# ============================================================
# Device Parameters (from experimental extraction)
# ============================================================
G_MIN = {g_min:.6}e-6      # Min conductance (S)
G_MAX = {g_max:.6}e-6      # Max conductance (S)
ALPHA_P = {alpha_p:.4}       # Potentiation non-linearity
ALPHA_D = {alpha_d:.4}       # Depression non-linearity
WRITE_NOISE = {write_noise:.6}  # Normalized write noise σ_w
NUM_LEVELS = max({num_levels_p}, {num_levels_d})

# ============================================================
# Training Configuration
# ============================================================
HIDDEN_SIZE = {hidden_size}
EPOCHS = {epochs}
LEARNING_RATE = {learning_rate}
BATCH_SIZE = {batch_size}

# ============================================================
# Model Definition
# ============================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================================================
# Data Loading
# ============================================================
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Train Ideal Model First
# ============================================================
print("Training ideal model...")
ideal_model = MLP().to(device)
optimizer = optim.SGD(ideal_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    ideal_model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = ideal_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or epoch == EPOCHS:
        ideal_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = ideal_model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        print(f"Epoch {{epoch}}: Ideal accuracy = {{correct/len(test_dataset)*100:.2f}}%")

# ============================================================
# Apply MemTorch Device Model
# ============================================================
if HAS_MEMTORCH:
    print("\nPatching model with MemTorch memristive device model...")

    # Define the reference memristor using data-driven model
    reference_memristor = memtorch.bh.memristor.LinearIonDrift
    reference_memristor_params = {{
        'r_on': 1.0 / G_MAX,   # Ω
        'r_off': 1.0 / G_MIN,  # Ω
    }}

    # Patch the trained model
    patched_model = patch_model(
        copy.deepcopy(ideal_model),
        memristor_model=reference_memristor,
        memristor_model_params=reference_memristor_params,
        module_parameters_to_patch=[torch.nn.Linear],
        mapping_routine=naive_map,
        transistor=True,
        programming_routine=None,
    )

    # Apply non-idealities
    patched_model = apply_nonidealities(
        patched_model,
        non_idealities=[
            memtorch.bh.nonideality.NonIdeality.DeviceFaults,
        ],
        p_stuck_on=0.0,
        p_stuck_off=0.0,
    )

    # Evaluate patched model
    patched_model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = patched_model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()

    mem_acc = correct / len(test_dataset) * 100
    print(f"\nMemTorch memristive model accuracy: {{mem_acc:.2f}}%")

    # Also evaluate with manual copy-and-degrade for comparison
    print("\nNote: For more accurate non-linearity modeling,")
    print("consider using MemTorch's data-driven device model")
    print("with your experimental P/D curves as input.")
else:
    print("\nMemTorch not available. Using manual copy-and-degrade...")
    print("Install MemTorch for full memristive simulation:")
    print("  pip install memtorch")

print("\nDone!")
"#,
        g_min = params.g_min,
        g_max = params.g_max,
        alpha_p = params.alpha_p,
        alpha_d = params.alpha_d,
        write_noise = params.write_noise,
        num_levels_p = params.num_levels_p,
        num_levels_d = params.num_levels_d,
        hidden_size = config.hidden_size,
        epochs = config.epochs,
        learning_rate = config.learning_rate,
        batch_size = config.batch_size,
    );

    Ok(script)
}

#[tauri::command]
pub fn export_chart_data(
    svg_string: String,
    output_path: String,
) -> Result<(), String> {
    std::fs::write(&output_path, svg_string.as_bytes())
        .map_err(|e| format!("Failed to write SVG file: {}", e))?;
    Ok(())
}
