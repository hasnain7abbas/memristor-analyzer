# Claude Code Instructions: Memristor Analyzer Updates

Copy this entire prompt and paste it into Claude Code in your `memristor-analyzer` repo directory.

---

## Prompt for Claude Code

```
I need you to implement several changes to this Tauri + React + Rust memristor analyzer app. Read through the full codebase first (especially src-tauri/src/parameters.rs, src-tauri/src/ann.rs, src-tauri/src/export.rs, and the React pages in src/pages/) before making any changes.

## CHANGE 1: Fix Alpha Fitting — Replace Grid Search with Proper Optimization (Rust backend)

In `src-tauri/src/parameters.rs`, the alpha (α) nonlinearity fitting currently uses the exponential model:

  G(n) = G_start + (G_end − G_start) · [1 − exp(−α·n/N)] / [1 − exp(−α)]

Replace this with the **power-law model** which is more standard and numerically stable:

  Potentiation: G(n) = G_min + ΔG · [1 − (1 − n/N)^α_P]
  Depression:   G(n) = G_min + ΔG · (1 − n/N)^α_D

For the fitting algorithm, replace any grid search with Brent's method (bounded scalar optimization) on the interval [0.01, 200]. Fit α with G_start and G_end as **free parameters** (not fixed to cross-cycle averages). This eliminates the grid-resolution artifact where α_D fluctuates between 30, 45, and 98 for abrupt depression curves.

If the codebase uses a grid search loop like `for a in (0.01..100.0).step(0.05)`, replace it entirely. Use a golden-section search or Brent's method implementation. Rust crate `argmin` can help, or implement a simple bounded minimization.

The key function signature should be something like:
```rust
fn fit_alpha_power_law(data: &[f64], is_depression: bool) -> (f64, f64, f64, f64)
// returns (alpha, r_squared, fitted_g_start, fitted_g_end)
```

Also update the model equation displayed on the Parameters page (in `src/pages/ParametersPage.tsx` and `src/components/FormulaCard.tsx` or wherever the KaTeX formula is rendered) to show the power-law model instead of the exponential.

## CHANGE 2: Fix the ANN Simulation (Rust backend + React frontend)

In `src-tauri/src/ann.rs`, the current ANN simulation has issues that make it unrealistically optimistic. Fix these:

1. **Architecture**: Use 784→100→10 (not 784→1500→10 or whatever is currently there). The hidden layer should be 100 neurons to match realistic crossbar array sizes.

2. **Activation**: Use sigmoid throughout (not ReLU). Real memristor crossbar simulations use sigmoid.

3. **No ideal bias bypass**: If biases are being trained with standard gradient descent while weights use memristor updates, either remove biases entirely or constrain them to memristor physics too. Real crossbar arrays don't have separately-trained digital biases.

4. **Sample-by-sample updates**: The weight update should use outer-product updates sample-by-sample (or at minimum, don't accumulate gradients over a full batch before applying pulse updates). Each training sample should trigger its own pulse update — this is how real crossbar training works.

5. **Update the copy-and-degrade method** to also use the power-law model (matching Change 1) for conductance quantization and noise injection.

## CHANGE 3: Update Export Scripts (Rust backend)

In `src-tauri/src/export.rs`, update the generated Python scripts:

### CrossSim export:
- Use the power-law model for LUT generation (matching the new fitting)
- The generated script should have a clear parameter block at the top with comments
- Architecture: 784→100→10, sigmoid, no bias
- Include the GPU monkey-patch for SciPy/CuPy as an optional block
- Include both CPU and GPU paths controlled by a USE_GPU flag
- At the top of the generated script, add a comment: "# Run this notebook in Google Colab for GPU access: upload this .py file or convert to .ipynb"

### PyTorch/MLP export:
- Match the CrossSim architecture: 784→100→10, sigmoid, no bias
- Use differential pair (G+, G-) for signed weights
- Sample-by-sample outer-product updates
- Power-law model for conductance curves
- Include both ideal baseline and memristor training in the same script

### MemTorch export:
- Either remove it entirely, or add a note at the top saying "MemTorch requires legacy Python 3.9 + PyTorch 1.10.0 environment. See installation instructions in the generated script."

### For ALL exported scripts:
- Add a clear "DEVICE PARAMETERS" section at the top with all values pre-filled from the app
- Add a "SIMULATION SETTINGS" section (hidden units, epochs, learning rate)
- Add a comment at the top: "# If GPU is needed, run on Google Colab: https://colab.research.google.com"
- Use the power-law alpha model consistently everywhere

## CHANGE 4: Make Parameter Input Easier (React frontend)

On the Upload page or Parameters page (wherever the user configures their data):

1. Ensure there are clear, labeled input fields for:
   - Number of cycles (N_CYCLES)
   - Pulses per phase (PPP) — support both 50 and 100
   - Read voltage (V_read) if current data needs conversion to conductance
   - Write/set voltage and reset voltage (optional, for documentation in exports)

2. These values should flow through to:
   - The parameter extraction (alpha fitting uses PPP)
   - The ANN simulation (number of conductance levels depends on PPP)
   - All export scripts (pre-filled in the parameter blocks)

3. The stride for splitting cycles should use `2 * PPP` (not hardcoded 100).

## CHANGE 5: Neural Network — Colab Redirect for GPU

In `src/pages/ANNPage.tsx` (or wherever the ANN training UI is):

1. Keep the in-app simulation for quick preview (Perceptron, small MLP)
2. For the full MNIST training that would benefit from GPU, add a prominent button/card that says:

   "🚀 Run Full Training on Google Colab"
   
   When clicked, it should:
   - Generate the Python script with all parameters pre-filled
   - Save it as a .py file
   - Show a dialog that says:
     "Full MNIST training with CrossSim requires GPU access. 
      1. Upload the exported script to Google Colab
      2. Go to Runtime → Change runtime type → GPU
      3. Run all cells
      
      [Open Google Colab](https://colab.research.google.com)"
   - Include a "Copy Colab link" button

3. For architectures that can't run in-app (LeNet-5, CNN), instead of trying to train them, show the Colab redirect immediately.

## Important Notes:
- The power-law model `w(n) = (1 - n/N)^α` should be used CONSISTENTLY everywhere — in Rust parameter extraction, in the ANN simulation, in all export scripts, and in the UI formula displays.
- Test that the stride `2 * PPP` works correctly when PPP is changed from 50 to 100.
- The alpha fitting MUST use bounded optimization (not grid search) with free G_start/G_end endpoints.
- Keep all existing features working — smoothing, export presets, chart gallery, etc.
```

---

## Quick Reference: What Goes Where

| Change | Files to modify |
|--------|----------------|
| Power-law alpha fitting | `src-tauri/src/parameters.rs` |
| ANN simulation fix | `src-tauri/src/ann.rs` |
| Export scripts | `src-tauri/src/export.rs` |
| Formula display | `src/pages/ParametersPage.tsx`, `src/components/FormulaCard.tsx`, `src/components/HelpDialog.tsx` |
| Parameter input fields | `src/pages/UploadPage.tsx` or `src/pages/ParametersPage.tsx` |
| Colab redirect | `src/pages/ANNPage.tsx` |
| Zustand store (if PPP/cycles need to flow through) | `src/stores/useAppStore.ts`, `src/types/index.ts` |
