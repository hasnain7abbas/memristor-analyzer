<p align="center">
  <img src="public/favicon.svg" width="140" height="140" alt="Memristor Neural Analyzer">
</p>

<h1 align="center">Memristor Neural Analyzer</h1>

<p align="center">
  <strong>Desktop application for memristor device characterization &amp; neuromorphic ANN simulation</strong>
</p>

<p align="center">
  <a href="#features">Features</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#how-it-works">How It Works</a> &nbsp;·&nbsp;
  <a href="#formula-reference">Formulas</a> &nbsp;·&nbsp;
  <a href="#tech-stack">Tech Stack</a> &nbsp;·&nbsp;
  <a href="#changelog">Changelog</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Tauri-2.10-blue?logo=tauri&logoColor=white" alt="Tauri 2.10">
  <img src="https://img.shields.io/badge/React-19-61dafb?logo=react&logoColor=white" alt="React 19">
  <img src="https://img.shields.io/badge/Rust-Backend-orange?logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/TypeScript-5.9-3178c6?logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

---

Upload experimental potentiation/depression data, smooth noisy curves, extract key device parameters (α, σ_w, G<sub>min</sub>/G<sub>max</sub>), simulate ANN training with realistic memristor non-idealities, and export publication-quality figures — all in one tool.

## Features

### 1. Data Upload
- **9 memristor test types** — P/D training, LTP/LTD, endurance, multilevel, EPSC, PPF, SDDP, SRDP, bipolar I-V
- CSV and Excel file parsing with **automatic column detection**
- Interactive axis selection and data preview charts
- Current-to-conductance auto-conversion (enter V<sub>read</sub>)
- **Demo data generator** — test the full pipeline instantly without measurement data

### 2. Smoothing
| Method | Best For |
|--------|----------|
| **Savitzky-Golay** | General purpose, preserves peak shapes (reflected boundary padding) |
| **LOESS/LOWESS** | Non-linear P/D patterns, locally weighted regression |
| **Gaussian kernel** | Gentle smoothing with shape preservation |
| **Median filter** | Removing single-point spike artifacts |
| **Moving average** | Simple noise reduction |

- **Monotonicity enforcement** via isotonic regression (pool adjacent violators)
- Adjustable strength blending (0–100% mix of raw and smoothed)
- Quality metrics: R² fit and shape preservation score

### 3. Parameter Extraction

All formulas are rendered with **KaTeX** for proper mathematical notation.

| Parameter | Formula | What It Tells You |
|-----------|---------|-------------------|
| **Conductance** | G = I<sub>read</sub> / V<sub>read</sub> | Basic device state |
| **On/Off Ratio** | G<sub>max</sub> / G<sub>min</sub> | Weight storage margin |
| **Non-linearity α** | G(n) = G<sub>start</sub> + (G<sub>end</sub> − G<sub>start</sub>) · [1 − e<sup>−αn/N</sup>] / [1 − e<sup>−α</sup>] | **The key parameter** — ANN accuracy impact |
| **Write Noise σ<sub>w</sub>** | σ(ΔG) / (G<sub>max</sub> − G<sub>min</sub>) | Weight update reproducibility |
| **CCV%** | σ(\|ΔG\|) / μ(\|ΔG\|) × 100% | Cycle-to-cycle variation |
| **Memory Window** | 20 · log₁₀(G<sub>max</sub>/G<sub>min</sub>) dB | Logarithmic On/Off measure |
| **Programming Margin** | (G<sub>max</sub> − G<sub>min</sub>) / (G<sub>max</sub> + G<sub>min</sub>) × 100% | HRS/LRS separation quality |
| **Asymmetry Index** | \|α<sub>P</sub> − α<sub>D</sub>\| / max(α<sub>P</sub>, α<sub>D</sub>) | P/D symmetry |
| **Switching Uniformity** | 1 − \|slope of ΔG vs G\| | Update uniformity (SUI) |
| **Distinguishable Levels** | ΔG / (μ<sub>step</sub> + 2σ<sub>step</sub>) | Bits per synapse |

- **Interactive Help button** — opens a comprehensive formula reference with exhaustive derivations, physical meaning, acquisition guides, quality thresholds, and literature references

### 4. ANN / MNIST Simulation

Train a neural network and see exactly how your device's non-idealities degrade accuracy:

- **Copy-and-degrade methodology** — ideal network trained with standard backpropagation, memristor version created by cloning weights and applying device non-idealities
- **Training algorithm** — SGD with momentum (0.9), cosine annealing learning rate, mini-batch gradient accumulation
- **Noise-averaged evaluation** — memristor accuracy averaged over multiple noise realizations per epoch for stable curves
- **Architectures**: Perceptron, MLP (1 hidden), MLP (2 hidden) in-app; LeNet-5 and CNN via Python export
- **Real-time chart** showing ideal vs memristor accuracy per epoch
- **Python script export** for 4 external frameworks:

| Framework | Description |
|-----------|-------------|
| **Custom PyTorch** | Full MNIST training with copy-and-degrade, all architectures |
| **CrossSim** (Sandia National Labs) | Analog crossbar array simulation |
| **NeuroSim** (Georgia Tech) | Device config + C++ parameter block for circuit-level benchmarking |
| **MemTorch** | PyTorch model patching with memristive device models |

### 5. Export
- **Publication presets** — Frontiers/Elsevier, Nature/Science, ACS, IEEE, presentation (16:9)
- Full chart customization (fonts, colors, line width, DPI, grid, borders)
- Chart gallery with all generated figures
- **SVG** and **PNG** export at configurable DPI (up to 600)

---

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) 22+
- [Rust](https://rustup.rs/) stable
- [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)

### Install & Run

```bash
git clone https://github.com/hasnain7abbas/memristor-analyzer.git
cd memristor-analyzer
npm install
npm run tauri dev
```

### Build Installer

```bash
npm run tauri build
```

Produces `.msi` and `.exe` installers in `src-tauri/target/release/bundle/`.

---

## How It Works

```
┌──────────┐     ┌───────────┐     ┌────────────┐     ┌──────────┐     ┌────────┐
│  Upload  │ ──▶ │ Smoothing │ ──▶ │ Parameters │ ──▶ │   ANN    │ ──▶ │ Export │
│          │     │           │     │            │     │          │     │        │
│ CSV/XLSX │     │ SG, LOESS │     │ α, σ_w,    │     │ Ideal vs │     │ SVG    │
│ 9 tests  │     │ Gaussian  │     │ Gmin, Gmax │     │ Memristor│     │ PNG    │
│ Demo data│     │ Monotonic │     │ CCV, SUI   │     │ Training │     │ Python │
└──────────┘     └───────────┘     └────────────┘     └──────────┘     └────────┘
```

1. **Upload** — Load your measurement CSV/Excel file (or use demo data). The tool auto-detects column types and splits potentiation/depression phases.

2. **Smoothing** — Apply noise reduction to raw P/D curves. Compare methods side-by-side with R² quality metrics.

3. **Parameters** — Extract α (non-linearity), σ<sub>w</sub> (write noise), G<sub>min</sub>/G<sub>max</sub>, and 10+ other parameters. All formulas are beautifully rendered with KaTeX. Click **Help** for exhaustive documentation.

4. **ANN / MNIST** — Run a neural network simulation using your extracted parameters. The ideal network shows perfect weight storage; the memristor network shows the accuracy drop from your device's non-idealities. Smooth, consistent convergence curves.

5. **Export** — Download publication-quality charts (SVG/PNG at up to 600 DPI) or generate Python scripts for PyTorch, CrossSim, NeuroSim, or MemTorch with your parameters pre-filled.

---

## Formula Reference

The app includes a comprehensive **Help** dialog (accessible from the nav bar) with:

- **KaTeX-rendered equations** — proper mathematical notation (fractions, subscripts, Greek letters)
- **Parameter tables** — every variable explained with units and physical meaning
- **How to acquire** — step-by-step experimental measurement instructions
- **Physical derivations** — from Ohm's law through non-linear kinetic equations
- **Quality thresholds** — what values are good/acceptable/poor for each parameter
- **ANN impact** — how each parameter affects neuromorphic computing performance
- **Literature references** — Chua 1971, Burr et al. 2017, Zucker & Regehr 2002

### Key Formula: Non-Linearity α

The central parameter extracted by this tool:

```
G(n) = G_start + (G_end − G_start) · [1 − exp(−α·n/N)] / [1 − exp(−α)]
```

### Literature-Calibrated Quality Table

Quality depends on **both** alpha and the conductance ratio (G<sub>max</sub>/G<sub>min</sub>). Calibrated against Kim et al. (2021), Burr et al. (2015), and published device benchmarks.

| α_P | α_D | G_max/G_min | Expected MNIST (MLP 784-300-10) | Quality Rating |
|-----|-----|-------------|--------------------------------|----------------|
| <0.5 | <0.5 | >10 | 93-97% | Excellent |
| 0.5-1.5 | 0.5-1.5 | >5 | 87-93% | Good |
| 1.5-3.0 | 1.5-3.0 | >3 | 75-87% | Acceptable |
| 3.0-5.0 | 3.0-5.0 | >2 | 55-75% | Degraded |
| >5.0 | >5.0 | any | <55% | Poor |
| any | any | <2 | deduct 10-20% | Window penalty |

**Key insight from literature:** The conductance ratio (G<sub>max</sub>/G<sub>min</sub>) has a multiplicative effect on accuracy. A device with α = 2 but G<sub>max</sub>/G<sub>min</sub> = 50 will significantly outperform a device with α = 2 but G<sub>max</sub>/G<sub>min</sub> = 3.

---

## Methodology

### Copy-and-Degrade ANN Simulation

The ANN simulation follows the **copy-and-degrade** methodology established by Burr et al. (2015):

1. **Train ideal network** — A standard MLP (784-256-10 by default) is trained on synthetic MNIST using SGD with cosine annealing learning rate. Weights are stored in floating-point precision.

2. **Clone weights** — After each training epoch, the ideal weights are cloned to create the memristor network.

3. **Non-linear quantization** — Each weight is normalized to [0, 1] and snapped to the nearest achievable conductance level. The achievable levels are non-linearly spaced according to the device's α parameter:
   ```
   G(k) = [1 − exp(−α · k/(N−1))] / [1 − exp(−α)]   for k = 0, 1, ..., N−1
   ```
   Potentiation levels (α_P) are used for weights in the upper half of the range; depression levels (α_D, computed as `1 − G_P(k)` sorted ascending) are used for the lower half. This asymmetric level spacing is what causes the accuracy degradation — with large α, most levels cluster near one end of the range, leaving poor resolution elsewhere.

4. **Noise injection** — Gaussian write noise σ_w is added to each quantized weight, modeling the stochastic variation in conductance programming.

5. **Evaluate** — The degraded network is evaluated on the test set. Results are averaged over 5 independent noise realizations per epoch for stable accuracy estimates.

### Multi-Cycle Data Processing

When experimental data contains multiple P/D cycles (e.g., 50 potentiation pulses + 50 depression pulses, repeated N times):

1. **Cycle segmentation** — Data is split into individual cycles using user-specified pulses-per-phase or automatic peak detection.

2. **Cycle averaging** — All P cycles are averaged position-by-position to produce one representative potentiation curve; same for depression. This is the standard approach in memristor literature for extracting representative device behavior.

3. **Write noise from cycle-to-cycle variation** — σ_w is computed from the per-position standard deviation across cycles, normalized by the conductance range: `σ_w = mean(σ[n]) / (G_max − G_min)`.

4. **Parameter extraction** — α is fitted separately for the averaged P and D curves using grid search (coarse 0.01-12.0 in steps of 0.05, refined ±0.3 in steps of 0.001).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop Framework | Tauri 2.10 |
| Frontend | React 19, TypeScript 5.9, Vite 7 |
| Styling | Tailwind CSS 4 |
| Math Rendering | KaTeX |
| Charts | Recharts |
| State Management | Zustand |
| Backend | Rust (ndarray, calamine, csv) |
| Icons | Lucide React |
| File Parsing | PapaParse, SheetJS (xlsx) |

---

## Project Structure

```
memristor-analyzer/
├── src/
│   ├── components/
│   │   ├── Chart.tsx              # Recharts wrapper with SVG/PNG export
│   │   ├── FormulaCard.tsx        # Expandable formula cards with KaTeX
│   │   ├── HelpDialog.tsx         # Comprehensive formula reference dialog
│   │   ├── Tex.tsx                # KaTeX rendering helper
│   │   ├── Layout.tsx             # Navigation with Help button
│   │   ├── StatCard.tsx           # Metric display with quality indicators
│   │   ├── ChartControls.tsx      # Chart label/type controls
│   │   ├── SmoothingControls.tsx  # Smoothing algorithm selector
│   │   ├── FileUploader.tsx       # File picker with column mapping
│   │   └── DataTable.tsx          # Tabular data preview
│   ├── pages/
│   │   ├── UploadPage.tsx         # 9 test types + demo data
│   │   ├── SmoothingPage.tsx      # Method comparison + quality metrics
│   │   ├── ParametersPage.tsx     # α extraction + formula cards
│   │   ├── ANNPage.tsx            # MNIST training + framework export
│   │   └── ExportPage.tsx         # Publication presets + chart gallery
│   ├── stores/useAppStore.ts      # Zustand global state
│   ├── types/index.ts             # TypeScript interfaces
│   └── lib/                       # File parser utilities
├── src-tauri/src/
│   ├── ann.rs                     # MLP with momentum, cosine annealing, copy-and-degrade
│   ├── parameters.rs              # α fitting, CCV, write noise, switching uniformity
│   ├── smoothing.rs               # SG, LOESS, Gaussian, median, MA, monotonic
│   ├── export.rs                  # Python script generators (4 frameworks)
│   ├── file_io.rs                 # CSV/Excel reading with metadata skipping
│   └── lib.rs                     # Tauri command registration
└── .github/workflows/
    ├── ci.yml                     # TypeScript + Rust checks on push/PR
    └── release.yml                # Auto-version bump + build + GitHub Release
```

---

## CI/CD

- **CI** runs on every push and PR — validates TypeScript compilation and Rust checks
- **Release** runs on push to `main` — auto-bumps patch version, builds Windows `.msi` + `.exe` installers, creates GitHub Release with artifacts attached

---

## Changelog

### v1.0.6 (latest)
- **Multi-cycle data parsing** — new cycle configuration UI (pulses per potentiation/depression phase) enables proper segmentation of multi-cycle P/D data. Cycles are averaged position-by-position to produce representative P and D curves, following standard memristor characterization methodology. Write noise (σ_w) is now computed from cycle-to-cycle position variation rather than step-size variation, matching the definition used in Burr et al. (2015).
- **Single-cycle split fix** — when auto-detection finds 0-1 peaks, the data is now split at the midpoint instead of at the max conductance point. The previous approach failed when potentiation saturates early (common for non-linear devices with large α).
- **ANN depression levels fix** — depression conductance levels in the copy-and-degrade simulation are now computed as `1 − G_P(k)` sorted ascending, correctly placing levels in the lower half of the conductance range. Previously, depression levels were identical to potentiation levels, which meant most achievable states clustered in the wrong half. This follows Burr et al. (2015) methodology.
- **Increased noise realizations** — memristor accuracy is now averaged over 5 independent noise realizations per epoch (up from 2), producing more stable and reproducible accuracy curves per Burr et al. (2015) recommendation.
- **Default hidden size increased** — MLP hidden layer default changed from 128 to 256 neurons for better alignment with the MLP 784-256-10 architecture used in Kim et al. (2021) benchmarks.
- **Literature-calibrated quality table** — the alpha quality thresholds now account for both α and the conductance ratio (G_max/G_min), calibrated against published device benchmarks including Kim et al. (2021), Burr et al. (2015/2017), and multiple BFO/RRAM device studies.
- **About dialog** — new About dialog with developer info, project description, and complete academic references (Burr 2017, Burr 2015, Kim 2021, Chua 1971, Strukov 2008, Zucker 2002).
- **ANN training overhaul** — mini-batch gradient accumulation (efficient pre-allocated buffers), SGD with momentum (0.9), cosine annealing learning rate schedule. Accuracy curves are now smooth and consistent instead of fluctuating randomly
- **KaTeX formula rendering** — all formulas on the Parameters page now display with proper mathematical notation (fractions, subscripts, superscripts, Greek letters)
- **Comprehensive Help dialog** — accessible from the nav bar, contains 8 sections with exhaustive formula derivations, parameter tables, acquisition guides, physical meaning, quality thresholds, and literature references
- **Larger, clearer graphs** — content area widened to 1400px, chart heights increased, better margins and font sizing for readability
- **New application logo** — memristor hysteresis curve with magnifying glass

### v1.0.5
- Bug fixes and stability improvements

### v1.0.4
- **Smart CSV/TSV parser** — automatically skips Keithley instrument metadata rows
- **Empty file detection** — clear error messages for empty/malformed datasets
- **Demo data generator** — "Load Demo Data" button for testing the full pipeline
- **Fixed script export** — file system permissions for Python script downloads
- **Chart rendering fixes** — disabled animation glitches during live ANN training

### v1.0.3
- **LOESS/LOWESS smoothing** — locally weighted regression
- **Gaussian kernel smoothing** — weighted moving average
- **Monotonicity enforcement** — isotonic regression
- **Strength blending** — adjustable raw/smoothed mix
- **Multi-framework export** — CrossSim, NeuroSim, MemTorch scripts
- **Chart gallery** on Export page
- **Enhanced formula reference** — physical derivations and literature references

### v1.0.2
- **Publication-ready graphs** — presets for Frontiers, Nature, ACS, IEEE
- **Copy-and-degrade ANN** — realistic memristor simulation
- **Editable chart labels** — custom axis labels, captions, plot types
- **CI/CD pipeline** — automated builds and GitHub Releases

### v1.0.1
- Full parameter extraction (α, CCV%, σ_w, memory window, asymmetry index)
- Savitzky-Golay smoothing with reflected boundary padding
- ANN/MNIST simulation with 3 architecture options
- PyTorch script export

### v1.0.0
- Initial release — data upload, smoothing, parameter extraction, ANN training, chart export

---

## References

- L. Chua, "Memristor -- The Missing Circuit Element," *IEEE Trans. Circuit Theory*, vol. 18, no. 5, pp. 507-519, 1971
- D. B. Strukov, G. S. Snider, D. R. Stewart, R. S. Williams, "The missing memristor found," *Nature*, vol. 453, pp. 80-83, 2008
- G. W. Burr et al., "Neuromorphic computing using non-volatile memory," *Advances in Physics: X*, vol. 2, no. 1, 2017
- G. W. Burr et al., "Experimental demonstration and tolerancing of a large-scale neural network (165,000 synapses) using phase-change memory as the synaptic weight element," *IEDM*, 2015
- S. Kim et al., "Spiking neural network with memristor synapses having non-linear weight update," *Front. Comput. Neurosci.* 15:646125, 2021
- R. Zucker & W. Regehr, "Short-term synaptic plasticity," *Annu. Rev. Physiol.*, vol. 64, pp. 355-405, 2002
- Ma et al., "Flexible BFO FTJ memristors," *Nano Research*, 2021
- Luo et al., "PEDOT:PSS/pentacene organic synaptic devices," *Front. Neurosci.*, 2022
- Soren & Prakash, "Cu/BFO/FTO resistive switching," *ACS Applied Electronic Materials*, 2022
- Peng et al., "HfO2/BFO/HfO2 trilayer memristors," *Adv. Funct. Mater.*, 2021
- CrossSim benchmark, Sandia National Laboratories, 2021

## License

MIT
