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

## Changelog

### v1.0.4 (latest)
- **Smart CSV/TSV parser** — automatically skips Keithley instrument metadata rows to find real column headers
- **Empty file detection** — clear error messages when data files have no rows or no numeric values, instead of silently showing nothing
- **Frontend data validation** — catches empty/malformed datasets before storing, shows diagnostic info (headers found, row count)
- **Demo data generator** — "Load Demo Data" button on Upload page lets you test the full pipeline without real measurement data
- **Fixed script export** — added missing file system permissions; Python/CrossSim/NeuroSim/MemTorch script downloads now work correctly
- **Chart rendering fixes** — disabled animation glitches during live ANN training, reduced dot clutter on line charts

### v1.0.3
- **LOESS/LOWESS smoothing** — locally weighted regression, best for non-linear P/D patterns
- **Gaussian kernel smoothing** — weighted moving average with better shape preservation
- **Monotonicity enforcement** — isotonic regression to enforce increasing/decreasing curves
- **Strength blending** — adjustable 0-100% mix of raw and smoothed data
- **Multi-framework export** — Python script generators for CrossSim, NeuroSim, MemTorch (in addition to PyTorch)
- **Chart gallery** on Export page with all available charts and individual SVG/PNG download
- **Enhanced formula reference** — physical derivations, literature references, worked examples

### v1.0.2
- **Publication-ready graphs** — presets for Frontiers, Nature, ACS, IEEE, presentation formats
- **Fixed ANN accuracy** — copy-and-degrade approach for realistic memristor simulation
- **Editable chart labels** — custom axis labels, captions, plot type selection per chart
- **CI/CD pipeline** — automated builds and GitHub Releases on push to main

### v1.0.1
- **Full parameter extraction** — G_min/G_max, On/Off ratio, non-linearity α, CCV%, write noise, memory window, asymmetry index
- **Savitzky-Golay smoothing** with reflected boundary padding
- **ANN/MNIST simulation** — in-app training with perceptron, MLP (1H), MLP (2H) architectures
- **PyTorch script export** with device parameters pre-filled

### v1.0.0
- Initial release — data upload, smoothing, parameter extraction, ANN training, chart export

## License

MIT
