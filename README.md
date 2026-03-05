# Memristor Neural Analyzer

Desktop application for memristor researchers — upload experimental data, smooth it, extract device parameters, run MNIST ANN simulations, and export publication-quality charts.

Built with **Tauri 2** + **React 19** + **TypeScript** + **Rust**.

## Features

### Data Upload
- Support for 9 memristor test types (P/D, LTP/LTD, endurance, multilevel, EPSC, PPF, SDDP, SRDP, bipolar I-V)
- CSV and Excel file parsing with automatic column detection
- Interactive X/Y axis selection for preview charts
- Current-to-conductance auto-conversion

### Smoothing
- **Savitzky-Golay** with reflected boundary padding (no edge discontinuity)
- **LOESS/LOWESS** — locally weighted regression, best for non-linear P/D patterns
- **Gaussian kernel** smoothing
- Moving average and median filter
- **Monotonicity enforcement** via isotonic regression (pool adjacent violators)
- Adjustable strength blending (0–100% mix of raw and smoothed)
- Quality metrics: R² fit and shape preservation score

### Parameter Extraction
- Non-linearity α (potentiation and depression) via grid search fitting
- Conductance range (G_min, G_max), On/Off ratio
- CCV%, write noise σ_w, multi-cycle support
- Memory window (dB), programming margin (%), asymmetry index
- Switching uniformity from ΔG vs G regression
- Distinguishable conductance levels (not just data point count)
- Detailed formula reference with physical derivations and literature references

### ANN / MNIST Simulation
- In-app training with synthetic MNIST (5,000 samples) using copy-and-degrade
- Perceptron, MLP (1 hidden), MLP (2 hidden) architectures
- LeNet-5 and CNN available via Python script export
- Real-time accuracy vs epoch chart (ideal vs memristor)
- Python script generators for 4 frameworks:
  - **Custom PyTorch** — full MNIST training with copy-and-degrade
  - **CrossSim** (Sandia National Labs) — crossbar array simulation
  - **NeuroSim** (Georgia Tech) — device config + C++ parameter block
  - **MemTorch** — PyTorch model patching with memristive devices

### Export
- Publication presets (Frontiers/Elsevier, Nature/Science, ACS, IEEE, presentation)
- Full chart style customization (fonts, colors, line width, DPI, grid, borders)
- Chart gallery with all generated charts and individual SVG/PNG download buttons
- SVG export with proper xmlns for compatibility
- PNG export at configurable DPI using data URI (no canvas tainting)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Tauri 2.10 |
| Frontend | React 19, TypeScript, Vite 7 |
| Styling | Tailwind CSS 4 |
| Charts | Recharts |
| State | Zustand |
| Backend | Rust (ndarray, calamine, csv) |
| Icons | Lucide React |
| File parsing | PapaParse, SheetJS (xlsx) |

## Development

### Prerequisites

- [Node.js](https://nodejs.org/) 22+
- [Rust](https://rustup.rs/) stable
- [Tauri CLI](https://v2.tauri.app/start/prerequisites/)

### Setup

```bash
npm install
npm run tauri dev
```

### Build

```bash
npm run tauri build
```

Produces `.msi` and `.exe` installers in `src-tauri/target/release/bundle/`.

## Project Structure

```
memristor-analyzer/
├── src/
│   ├── components/        # Chart, FormulaCard, SmoothingControls, etc.
│   ├── pages/             # Upload, Smoothing, Parameters, ANN, Export
│   ├── stores/            # Zustand store (useAppStore)
│   ├── types/             # TypeScript interfaces
│   └── lib/               # File parser utilities
├── src-tauri/
│   └── src/
│       ├── file_io.rs     # CSV/Excel file reading
│       ├── smoothing.rs   # SG, LOESS, Gaussian, monotonic, median, MA
│       ├── parameters.rs  # α fitting, CCV, write noise, num_levels
│       ├── ann.rs         # MLP training with copy-and-degrade
│       └── export.rs      # Python script generators (PyTorch, CrossSim, NeuroSim, MemTorch)
└── .github/workflows/
    ├── ci.yml             # TypeScript + Rust checks on push/PR
    └── release.yml        # Auto-version bump + build + GitHub Release
```

## CI/CD

- **CI** runs on every push and PR — validates TypeScript compilation and Rust checks
- **Release** runs on push to main — auto-bumps patch version, builds Windows installers, creates GitHub Release with `.msi` and `.exe` attached

## License

MIT
