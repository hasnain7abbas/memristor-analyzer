import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useAppStore } from '../stores/useAppStore';
import { FormulaCard } from '../components/FormulaCard';
import { StatCard } from '../components/StatCard';
import { AppLineChart } from '../components/Chart';
import { ChartControls } from '../components/ChartControls';
import { Download } from 'lucide-react';
import { exportChartData, exportMultiSection } from '../lib/chartExport';
import type { ChartLocalSettings, FormulaDefinition } from '../types';

const FORMULAS: FormulaDefinition[] = [
  {
    title: 'Step 1: Current → Conductance',
    formula: 'G = I_read / V_read',
    latex: 'G = \\frac{I_{\\text{read}}}{V_{\\text{read}}}',
    explanation:
      'If your Keithley records current (µA) at a fixed read voltage, divide to get conductance in µS (microsiemens).',
    example:
      'I_read = 50 µA, V_read = 0.1 V\nG = 50 / 0.1 = 500 µS',
    sections: [
      {
        subtitle: "Ohm's Law Derivation",
        content: 'V = IR → I = GV → G = I/V',
        latex: 'V = IR \\;\\Rightarrow\\; I = GV \\;\\Rightarrow\\; G = \\frac{I}{V} = \\frac{1}{R}',
      },
      {
        subtitle: 'Unit Conversion Table',
        content: 'A → µA:  ×10⁶    |  S → µS:  ×10⁶\nA → mA:  ×10³    |  S → mS:  ×10³\nA → nA:  ×10⁹    |  S → nS:  ×10⁹',
      },
      {
        subtitle: 'Contact Resistance Note',
        content: 'G_measured = G_device + G_contact\nFor 2-probe: subtract contact resistance\nFor 4-probe (Kelvin): G_measured ≈ G_device',
      },
    ],
    physicalMeaning: 'Conductance represents the ease of charge carrier flow through the memristive filament. In filamentary memristors, G is proportional to the cross-sectional area of the conductive filament.',
    reference: 'L. Chua, "Memristor — The Missing Circuit Element," IEEE Trans. Circuit Theory, 1971',
  },
  {
    title: 'Step 2: G_min, G_max, On/Off Ratio',
    formula: 'On/Off = G_max / G_min, ΔG = G_max - G_min',
    latex: '\\text{On/Off} = \\frac{G_{\\max}}{G_{\\min}}, \\quad \\Delta G = G_{\\max} - G_{\\min}',
    explanation:
      'The conductance window between LRS and HRS. Larger windows give more analog levels for the ANN.',
    example: 'G_min = 10 µS, G_max = 100 µS\nOn/Off = 10.0, ΔG = 90 µS',
    sections: [
      {
        subtitle: 'Memory Window (dB)',
        content: 'MW = 20·log₁₀(G_max / G_min)',
        latex: '\\text{MW} = 20 \\cdot \\log_{10}\\!\\left(\\frac{G_{\\max}}{G_{\\min}}\\right)',
      },
      {
        subtitle: 'Number of Distinguishable Levels',
        content: 'N = ΔG / (mean_step + 2·σ_step)',
        latex: 'N = \\frac{\\Delta G}{\\bar{\\Delta G}_{\\text{step}} + 2\\,\\sigma_{\\text{step}}}',
      },
      {
        subtitle: 'Programming Margin',
        content: 'PM = (G_max - G_min) / (G_max + G_min) × 100%',
        latex: '\\text{PM} = \\frac{G_{\\max} - G_{\\min}}{G_{\\max} + G_{\\min}} \\times 100\\%',
      },
    ],
    physicalMeaning: 'G_min corresponds to the high-resistance state (HRS, thin/ruptured filament) and G_max to the low-resistance state (LRS, thick/connected filament). The On/Off ratio determines the signal-to-noise margin for weight storage.',
  },
  {
    title: 'Step 3: Non-Linearity α — THE KEY PARAMETER',
    formula: 'G(n) = G_start + (G_end - G_start) × [1 - exp(-α·n/N)] / [1 - exp(-α)]',
    latex: 'G(n) = G_{\\text{start}} + \\bigl(G_{\\text{end}} - G_{\\text{start}}\\bigr) \\cdot \\frac{1 - e^{-\\alpha \\cdot n/N}}{1 - e^{-\\alpha}}',
    explanation:
      'α quantifies how the conductance update "saturates" as the device approaches its limit. Lower α means more uniform weight updates, which is better for training ANNs.',
    example:
      'If α_P = 1.5 and α_D = 2.0:\nMild non-linearity. Expect ~5-15% accuracy drop vs ideal.',
    highlight: true,
    sections: [
      {
        subtitle: 'Physical Model Derivation',
        content: 'Kinetic equation: dG/dn = β·(G_end - G)^γ',
        latex: '\\frac{dG}{dn} = \\beta \\cdot (G_{\\text{end}} - G)^{\\gamma}',
      },
      {
        subtitle: 'Impact on ANN Accuracy',
        content: 'α < 1:  < 5% accuracy drop (excellent)\nα = 1-2: 5-15% drop (acceptable)\nα = 2-4: 15-30% drop (degraded)\nα > 4:  > 30% drop (poor, needs compensation)',
      },
      {
        subtitle: 'Asymmetry Index',
        content: 'AI = |α_P - α_D| / max(α_P, α_D)',
        latex: '\\text{AI} = \\frac{|\\alpha_P - \\alpha_D|}{\\max(\\alpha_P,\\, \\alpha_D)}',
      },
    ],
    physicalMeaning: 'The non-linearity arises from the electric field distribution within the switching layer. As the filament grows (potentiation), the field concentrates at the tip causing rapid initial growth that saturates. Depression non-linearity comes from the reverse process of filament dissolution.',
    reference: 'G. W. Burr et al., "Neuromorphic computing using non-volatile memory," Advances in Physics: X, 2017',
  },
  {
    title: 'Step 4: CCV & Write Noise',
    formula: 'CCV% = std(|ΔG|) / mean(|ΔG|) × 100%',
    latex: '\\text{CCV}\\% = \\frac{\\sigma(|\\Delta G|)}{\\mu(|\\Delta G|)} \\times 100\\%, \\quad \\sigma_w = \\frac{\\sigma(\\Delta G)}{G_{\\max} - G_{\\min}}',
    explanation:
      'Cycle-to-cycle variation (CCV) measures how reproducible each weight update is. Write noise σ_w is the normalized standard deviation of conductance changes.',
    example:
      'mean(|ΔG|) = 2.0 µS, std(|ΔG|) = 0.5 µS\nCCV = 0.5/2.0 × 100 = 25%\nσ_w = 0.5/90 = 0.0056',
    sections: [
      {
        subtitle: 'Statistical Derivation',
        content: 'CCV from coefficient of variation:',
        latex: 'P(\\Delta G) \\sim \\mathcal{N}(\\mu_{\\Delta G},\\, \\sigma^2_{\\Delta G}), \\quad \\text{CCV} = \\frac{\\sigma_{\\Delta G}}{\\mu_{\\Delta G}}',
      },
      {
        subtitle: 'Device Area Scaling',
        content: 'σ_w ∝ 1/√A  (A = device area)',
        latex: '\\sigma_w \\propto \\frac{1}{\\sqrt{A}}',
      },
      {
        subtitle: 'Multi-Cycle Formula',
        content: 'CCV_total = √(CCV²_intra + CCV²_inter)',
        latex: '\\text{CCV}_{\\text{total}} = \\sqrt{\\text{CCV}_{\\text{intra}}^2 + \\text{CCV}_{\\text{inter}}^2}',
      },
    ],
    physicalMeaning: 'Write noise originates from the stochastic nature of ion migration and filament formation/dissolution. Each write pulse causes a slightly different atomic rearrangement, leading to conductance variations.',
  },
  {
    title: 'Step 5: ΔG vs G Scatter Plot',
    formula: 'x = G[i], y = G[i+1] - G[i]',
    latex: 'x = G_i, \\quad y = G_{i+1} - G_i',
    explanation:
      'This scatter plot reveals the state-dependent switching behavior. Ideally, ΔG should be constant (horizontal line). A slope indicates non-linearity.',
    sections: [
      {
        subtitle: 'State-Dependent Model',
        content: 'Potentiation: ΔG(G) = α_eff · (G_max - G)\nDepression: ΔG(G) = -α_eff · (G - G_min)',
        latex: '\\Delta G_{\\text{pot}}(G) = \\alpha_{\\text{eff}} \\cdot (G_{\\max} - G), \\quad \\Delta G_{\\text{dep}}(G) = -\\alpha_{\\text{eff}} \\cdot (G - G_{\\min})',
      },
      {
        subtitle: 'Switching Uniformity Index',
        content: 'SUI = 1 - |slope of ΔG vs G regression|',
        latex: '\\text{SUI} = 1 - \\bigl|\\text{slope of } \\Delta G \\text{ vs } G \\text{ regression}\\bigr|',
      },
    ],
    physicalMeaning: 'A negative slope in ΔG vs G for potentiation means the device updates less as it approaches G_max (saturation behavior). This is the physical manifestation of non-linearity α.',
  },
  {
    title: 'Step 6: PPF Index',
    formula: 'PPF Index = (A2 - A1) / A1 × 100%',
    latex: '\\text{PPF} = \\frac{A_2 - A_1}{A_1} \\times 100\\%',
    explanation:
      'Paired-pulse facilitation measures short-term synaptic plasticity. τ₁ and τ₂ correspond to fast and slow decay timescales.',
    sections: [
      {
        subtitle: 'Full Double Exponential Model',
        content: 'PPF(Δt) = 1 + C₁·exp(-Δt/τ₁) + C₂·exp(-Δt/τ₂)',
        latex: '\\text{PPF}(\\Delta t) = 1 + C_1 \\, e^{-\\Delta t / \\tau_1} + C_2 \\, e^{-\\Delta t / \\tau_2}',
      },
      {
        subtitle: 'Biological Comparison',
        content: 'Hippocampal synapse:\n  τ₁ ≈ 50 ms, τ₂ ≈ 500 ms\n\nMemristor (typical):\n  τ₁ ≈ 100-500 ms, τ₂ ≈ 2-20 s',
      },
    ],
    physicalMeaning: 'PPF in memristors arises from residual ion concentration near the filament tip after the first pulse. The second pulse encounters a pre-conditioned switching layer, requiring less energy for the same conductance change.',
    reference: 'R. Zucker & W. Regehr, "Short-term synaptic plasticity," Annu. Rev. Physiol., 2002',
  },
];

/**
 * Auto-detect potentiation/depression cycles from raw conductance data.
 * Handles both single-sweep (one P then one D) and multi-cycle data
 * (e.g., 50 pulses up, 50 pulses down, repeated N times).
 *
 * Uses derivative sign-change detection instead of peak finding, which is
 * robust to any cycle period (the old peak-window approach failed when
 * the window was larger than the cycle period).
 */
function autoDetectPDCycles(values: number[]): { potentiation: number[]; depression: number[]; detectedPulsesPerP: number; detectedPulsesPerD: number } {
  const n = values.length;
  if (n < 10) return { potentiation: values, depression: [], detectedPulsesPerP: n, detectedPulsesPerD: 0 };

  // Step 1: Light smoothing for derivative computation (window=5)
  const smoothed: number[] = [];
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - 2);
    const hi = Math.min(n, i + 3);
    let sum = 0;
    for (let j = lo; j < hi; j++) sum += values[j];
    smoothed.push(sum / (hi - lo));
  }

  // Step 2: Compute derivative signs
  const signs: number[] = [];
  for (let i = 1; i < n; i++) {
    signs.push(Math.sign(smoothed[i] - smoothed[i - 1]));
  }

  // Step 3: Find sign changes (positive→negative = peak, negative→positive = valley)
  // Filter out zero-derivative regions by looking at the last non-zero sign.
  const transitions: { index: number; type: 'peak' | 'valley' }[] = [];
  let lastSign = 0;
  for (let i = 0; i < signs.length; i++) {
    if (signs[i] === 0) continue;
    if (lastSign > 0 && signs[i] < 0) {
      transitions.push({ index: i, type: 'peak' });
    } else if (lastSign < 0 && signs[i] > 0) {
      transitions.push({ index: i, type: 'valley' });
    }
    lastSign = signs[i];
  }

  // Step 4: Filter out spurious transitions that are too close together
  // (noise-induced micro-reversals). Minimum distance = 10 points.
  const filtered: typeof transitions = [];
  for (const t of transitions) {
    if (filtered.length === 0 || t.index - filtered[filtered.length - 1].index >= 10) {
      // Also require alternating peak/valley
      if (filtered.length === 0 || filtered[filtered.length - 1].type !== t.type) {
        filtered.push(t);
      }
    }
  }

  // Step 5: If we have multiple cycles, estimate period and segment
  if (filtered.length >= 3) {
    // Estimate half-period from median spacing between consecutive transitions
    const spacings: number[] = [];
    for (let i = 1; i < filtered.length; i++) {
      spacings.push(filtered[i].index - filtered[i - 1].index);
    }
    spacings.sort((a, b) => a - b);
    const halfPeriod = spacings[Math.floor(spacings.length / 2)];
    const period = halfPeriod * 2;

    // Determine which half is P and which is D by checking the first transition
    const firstRising = filtered[0].type === 'peak'; // first segment was rising = potentiation

    // Segment into P and D blocks using the detected period
    const pulsesPerHalf = halfPeriod;
    const cycleLen = period;
    const numCycles = Math.floor(n / cycleLen);

    if (numCycles >= 1) {
      const pCycles: number[][] = [];
      const dCycles: number[][] = [];

      // Find the start: look for the first valley (start of potentiation)
      // or use index 0 if data starts with potentiation
      let dataStart = 0;
      if (filtered.length > 0 && filtered[0].type === 'valley' && filtered[0].index < halfPeriod) {
        dataStart = filtered[0].index;
      }

      for (let c = 0; c < numCycles; c++) {
        const start = dataStart + c * cycleLen;
        const mid = start + pulsesPerHalf;
        const end = start + cycleLen;
        if (end <= n) {
          if (firstRising) {
            pCycles.push(values.slice(start, mid));
            dCycles.push(values.slice(mid, end));
          } else {
            dCycles.push(values.slice(start, mid));
            pCycles.push(values.slice(mid, end));
          }
        }
      }

      if (pCycles.length >= 1 && dCycles.length >= 1) {
        return {
          potentiation: averageAlignedSegments(pCycles),
          depression: averageAlignedSegments(dCycles),
          detectedPulsesPerP: pCycles[0]?.length ?? halfPeriod,
          detectedPulsesPerD: dCycles[0]?.length ?? halfPeriod,
        };
      }
    }
  }

  // Fallback: single-sweep — split at midpoint
  const mid = Math.floor(n / 2);
  return {
    potentiation: values.slice(0, mid),
    depression: values.slice(mid),
    detectedPulsesPerP: mid,
    detectedPulsesPerD: n - mid,
  };
}

/** Average multiple segments of possibly different lengths into one representative curve. */
function averageAlignedSegments(segments: number[][]): number[] {
  if (segments.length === 0) return [];
  if (segments.length === 1) return segments[0];

  // Use the median length as the target
  const lengths = segments.map(s => s.length).sort((a, b) => a - b);
  const targetLen = lengths[Math.floor(lengths.length / 2)];

  // Linearly interpolate each segment to targetLen, then average
  const result: number[] = new Array(targetLen).fill(0);
  for (const seg of segments) {
    for (let i = 0; i < targetLen; i++) {
      // Map index i in [0, targetLen-1] to position in segment
      const pos = (i / (targetLen - 1)) * (seg.length - 1);
      const lo = Math.floor(pos);
      const hi = Math.min(lo + 1, seg.length - 1);
      const frac = pos - lo;
      result[i] += seg[lo] * (1 - frac) + seg[hi] * frac;
    }
  }
  for (let i = 0; i < targetLen; i++) result[i] /= segments.length;

  return result;
}

export function ParametersPage() {
  const { uploadedTests, smoothingConfig, extractedParams, setExtractedParams, vRead, setVRead, isCurrentInput, cycleConfig, setCycleConfig } =
    useAppStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [pdChart, setPdChart] = useState<ChartLocalSettings>({
    xLabel: 'Pulse Number', yLabel: 'Conductance (µS)', plotType: 'line', caption: '',
  });
  const [dgChart, setDgChart] = useState<ChartLocalSettings>({
    xLabel: 'G (µS)', yLabel: 'ΔG (µS)', plotType: 'scatter', caption: '',
  });

  const pdTest = uploadedTests['pd_training'] || uploadedTests['ltp_ltd'];

  const handleExtract = async () => {
    if (!pdTest) return;
    setLoading(true);
    setError(null);

    try {
      const ds = pdTest.dataset;
      const mapping = pdTest.columnMapping;

      const condColName = mapping['conductance_uS'] || ds.headers.find((h) => /conductance|cond|g_?us|^g$/i.test(h));
      const currentColName = mapping['current_uA'] || ds.headers.find((h) => /current|i_?ua/i.test(h));
      const colName = condColName || currentColName;

      if (!colName) {
        throw new Error('Cannot find conductance or current column. Please map columns on the Upload tab.');
      }

      const values: number[] = ds.rows
        .map((r) => {
          const v = r[colName];
          return typeof v === 'number' ? v : parseFloat(String(v));
        })
        .filter((v) => !isNaN(v));

      if (values.length < 4) {
        throw new Error('Need at least 4 data points');
      }

      const typeColName = mapping['type'] || ds.headers.find((h) => /type|phase/i.test(h));
      let pRaw: number[];
      let dRaw: number[];
      let computedWriteNoise: number | null = null;

      // Helper: segment values using pulsesPerP / pulsesPerD and average across cycles.
      // Also computes write noise (sigma_w) from per-position std across cycles.
      const segmentByCycleConfig = (vals: number[]): { p: number[]; d: number[]; sigmaW: number | null } => {
        const cycleLen = cycleConfig.pulsesPerP + cycleConfig.pulsesPerD;
        const numCycles = Math.floor(vals.length / cycleLen);
        if (numCycles < 1) {
          // Not enough data for even one cycle; split at midpoint
          const mid = Math.floor(vals.length / 2);
          return { p: vals.slice(0, mid), d: vals.slice(mid), sigmaW: null };
        }
        const pCycles: number[][] = [];
        const dCycles: number[][] = [];
        for (let c = 0; c < numCycles; c++) {
          const start = c * cycleLen;
          pCycles.push(vals.slice(start, start + cycleConfig.pulsesPerP));
          dCycles.push(vals.slice(start + cycleConfig.pulsesPerP, start + cycleLen));
        }
        const pAvg = averageAlignedSegments(pCycles);
        const dAvg = averageAlignedSegments(dCycles);

        // Compute sigma_w from per-position std across cycles
        let sigmaW: number | null = null;
        if (numCycles > 1) {
          let sumStd = 0;
          let countPositions = 0;
          // Compute per-position std across P cycles
          for (const cycles of [pCycles, dCycles]) {
            const len = cycles[0]?.length ?? 0;
            for (let pos = 0; pos < len; pos++) {
              const posVals = cycles.map(c => c[pos]).filter(v => v !== undefined);
              if (posVals.length > 1) {
                const mean = posVals.reduce((a, b) => a + b, 0) / posVals.length;
                const variance = posVals.reduce((a, v) => a + (v - mean) ** 2, 0) / (posVals.length - 1);
                sumStd += Math.sqrt(variance);
                countPositions++;
              }
            }
          }
          if (countPositions > 0) {
            const allVals = [...pAvg, ...dAvg];
            const gMin = Math.min(...allVals);
            const gMax = Math.max(...allVals);
            const range = gMax - gMin;
            if (range > 1e-15) {
              sigmaW = (sumStd / countPositions) / range;
            }
          }
        }
        return { p: pAvg, d: dAvg, sigmaW };
      };

      if (typeColName) {
        const typeVals = ds.rows.map((r) => String(r[typeColName] || '').toLowerCase());
        const allP = values.filter((_, i) => /^p|pot|ltp/i.test(typeVals[i]));
        const allD = values.filter((_, i) => /^d|dep|ltd/i.test(typeVals[i]));

        if (allP.length === 0 || allD.length === 0) {
          // Phase column exists but no recognized labels; use cycle config or midpoint
          if (!cycleConfig.autoDetect) {
            const seg = segmentByCycleConfig(values);
            pRaw = seg.p;
            dRaw = seg.d;
            computedWriteNoise = seg.sigmaW;
          } else {
            const mid = Math.floor(values.length / 2);
            pRaw = values.slice(0, mid);
            dRaw = values.slice(mid);
          }
        } else {
          // Detect multi-cycle: find where phase transitions happen (P->D or D->P).
          const pCycles: number[][] = [];
          const dCycles: number[][] = [];
          let currentPhase = typeVals[0];
          let currentBlock: number[] = [];

          for (let i = 0; i < values.length; i++) {
            const phase = typeVals[i];
            if (!(/^p|pot|ltp/i.test(phase) || /^d|dep|ltd/i.test(phase))) continue;
            if (phase !== currentPhase && currentBlock.length > 0) {
              if (/^p|pot|ltp/i.test(currentPhase)) pCycles.push(currentBlock);
              else if (/^d|dep|ltd/i.test(currentPhase)) dCycles.push(currentBlock);
              currentBlock = [];
              currentPhase = phase;
            }
            currentBlock.push(values[i]);
          }
          if (currentBlock.length > 0) {
            if (/^p|pot|ltp/i.test(currentPhase)) pCycles.push(currentBlock);
            else if (/^d|dep|ltd/i.test(currentPhase)) dCycles.push(currentBlock);
          }

          if (pCycles.length > 1 || dCycles.length > 1) {
            pRaw = averageAlignedSegments(pCycles);
            dRaw = averageAlignedSegments(dCycles);
            // Compute sigma_w from cycle-to-cycle variation
            let sumStd = 0;
            let countPositions = 0;
            for (const cycles of [pCycles, dCycles]) {
              const len = Math.min(...cycles.map(c => c.length));
              for (let pos = 0; pos < len; pos++) {
                const posVals = cycles.map(c => c[pos]);
                if (posVals.length > 1) {
                  const mean = posVals.reduce((a, b) => a + b, 0) / posVals.length;
                  const variance = posVals.reduce((a, v) => a + (v - mean) ** 2, 0) / (posVals.length - 1);
                  sumStd += Math.sqrt(variance);
                  countPositions++;
                }
              }
            }
            if (countPositions > 0) {
              const allAvg = [...pRaw, ...dRaw];
              const gMin = Math.min(...allAvg);
              const gMax = Math.max(...allAvg);
              const range = gMax - gMin;
              if (range > 1e-15) {
                computedWriteNoise = (sumStd / countPositions) / range;
              }
            }
          } else {
            pRaw = allP;
            dRaw = allD;
          }
        }
      } else if (!cycleConfig.autoDetect) {
        // User-specified cycle structure: segment by pulsesPerP / pulsesPerD
        const seg = segmentByCycleConfig(values);
        pRaw = seg.p;
        dRaw = seg.d;
        computedWriteNoise = seg.sigmaW;
      } else {
        // Auto-detect P/D cycles by finding local peaks and valleys.
        const detected = autoDetectPDCycles(values);
        pRaw = detected.potentiation;
        dRaw = detected.depression;
      }

      const cycleColName = mapping['cycle'] || ds.headers.find((h) => /cycle/i.test(h));
      let multiCycleData: number[][] | null = null;
      if (cycleColName) {
        const cycleMap = new Map<string, number[]>();
        ds.rows.forEach((r) => {
          const cycle = String(r[cycleColName] || '0');
          const val = typeof r[colName] === 'number' ? r[colName] as number : parseFloat(String(r[colName]));
          if (!isNaN(val)) {
            if (!cycleMap.has(cycle)) cycleMap.set(cycle, []);
            cycleMap.get(cycle)!.push(val);
          }
        });
        if (cycleMap.size > 1) {
          multiCycleData = Array.from(cycleMap.values());
        }
      }

      const result = await invoke<any>('extract_parameters', {
        potentiationRaw: pRaw,
        depressionRaw: dRaw,
        smoothingConfig: {
          method: smoothingConfig.method,
          window_size: smoothingConfig.windowSize,
          poly_order: smoothingConfig.method === 'savitzky_golay' ? smoothingConfig.polyOrder : null,
          remove_outliers: smoothingConfig.removeOutliers,
          outlier_sigma: smoothingConfig.outlierSigma,
        },
        vRead,
        isCurrent: isCurrentInput && !!currentColName,
        multiCycleData,
      });

      // If we computed a better write noise from cycle-to-cycle variation, override
      if (computedWriteNoise !== null && computedWriteNoise > 0) {
        result.writeNoise = computedWriteNoise;
      }
      setExtractedParams(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const rQuality = (r2: number) => (r2 >= 0.95 ? 'good' : r2 >= 0.85 ? 'ok' : 'poor');

  // Literature-calibrated quality assessment: combined alpha + conductance ratio
  // Based on Kim et al. 2021, Burr et al. 2015, Burr et al. 2017
  const alphaQuality = (alpha: number): 'good' | 'ok' | 'poor' => {
    const ratio = extractedParams ? extractedParams.onOffRatio : 10;
    // Window penalty: low conductance ratio degrades quality
    const penalty = ratio < 2 ? 2 : ratio < 3 ? 1 : 0;
    const effectiveAlpha = alpha + penalty;
    if (effectiveAlpha < 1.5 && ratio > 5) return 'good';
    if (effectiveAlpha < 3.0 && ratio > 3) return 'ok';
    return 'poor';
  };

  // Build chart data
  const pdChartData = extractedParams
    ? [
        ...extractedParams.potentiationSmoothed.map((v: number, i: number) => ({
          index: i + 1,
          potSmoothed: v as number | undefined,
          potFitted: extractedParams.potentiationFitted[i] as number | undefined,
          potRaw: extractedParams.potentiationRaw[i] as number | undefined,
          depSmoothed: undefined as number | undefined,
          depFitted: undefined as number | undefined,
          depRaw: undefined as number | undefined,
        })),
        ...extractedParams.depressionSmoothed.map((v: number, i: number) => ({
          index: extractedParams.potentiationSmoothed.length + i + 1,
          potSmoothed: undefined as number | undefined,
          potFitted: undefined as number | undefined,
          potRaw: undefined as number | undefined,
          depSmoothed: v as number | undefined,
          depFitted: extractedParams.depressionFitted[i] as number | undefined,
          depRaw: extractedParams.depressionRaw[i] as number | undefined,
        })),
      ]
    : [];

  // Reshape deltaG data from {x,y}[] to keyed records for AppLineChart
  const deltaGData = extractedParams
    ? extractedParams.deltaG.map((d: { G: number; dG: number }) => ({ G: d.G, dG: d.dG }))
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text mb-1">Parameter Extraction</h2>
        <p className="text-sm text-text-muted">
          Extract memristor device parameters for ANN simulation.
        </p>
      </div>

      {/* Formula reference */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-text-muted">Formula Reference</h3>
        {FORMULAS.map((f, i) => (
          <FormulaCard key={i} {...f} />
        ))}
      </div>

      {/* Extraction controls */}
      <div className="bg-surface rounded-xl border border-border p-4 space-y-4">
        <div className="flex items-end gap-4 flex-wrap">
          <div>
            <label className="block text-xs text-text-muted mb-1">V_read (V)</label>
            <input
              type="number"
              step={0.01}
              value={vRead}
              onChange={(e) => setVRead(parseFloat(e.target.value) || 0.1)}
              className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text w-24 font-mono"
            />
          </div>
          <button
            onClick={handleExtract}
            disabled={loading || !pdTest}
            className="px-6 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent/80 disabled:opacity-50"
          >
            {loading ? 'Extracting...' : 'Extract Parameters'}
          </button>
          {!pdTest && (
            <span className="text-xs text-text-dim">
              Upload P/D Training or LTP/LTD data first
            </span>
          )}
        </div>

        {/* Cycle Configuration */}
        <div className="border-t border-border/50 pt-3">
          <div className="flex items-center gap-3 mb-3">
            <h4 className="text-sm font-medium text-text">Cycle Structure</h4>
            <label className="flex items-center gap-1.5 text-xs text-text-muted cursor-pointer">
              <input
                type="checkbox"
                checked={cycleConfig.autoDetect}
                onChange={(e) => setCycleConfig({ autoDetect: e.target.checked })}
                className="accent-accent"
              />
              Auto-detect
            </label>
          </div>
          {!cycleConfig.autoDetect && (
            <div className="flex items-end gap-4">
              <div>
                <label className="block text-xs text-text-muted mb-1">Pulses per Potentiation</label>
                <input
                  type="number"
                  min={1}
                  value={cycleConfig.pulsesPerP}
                  onChange={(e) => setCycleConfig({ pulsesPerP: Math.max(1, parseInt(e.target.value) || 50) })}
                  className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text w-28 font-mono"
                />
              </div>
              <div>
                <label className="block text-xs text-text-muted mb-1">Pulses per Depression</label>
                <input
                  type="number"
                  min={1}
                  value={cycleConfig.pulsesPerD}
                  onChange={(e) => setCycleConfig({ pulsesPerD: Math.max(1, parseInt(e.target.value) || 50) })}
                  className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text w-28 font-mono"
                />
              </div>
              {pdTest && (
                <span className="text-xs text-text-dim">
                  {Math.floor(
                    (pdTest.dataset.rows.length) / (cycleConfig.pulsesPerP + cycleConfig.pulsesPerD)
                  )}{' '}
                  cycle(s) detected from {pdTest.dataset.rows.length} rows
                </span>
              )}
            </div>
          )}
        </div>

        {error && <p className="text-sm text-red">{error}</p>}
      </div>

      {/* Results */}
      {extractedParams && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard label="G_min" value={extractedParams.Gmin.toFixed(4)} unit="µS" />
            <StatCard label="G_max" value={extractedParams.Gmax.toFixed(4)} unit="µS" />
            <StatCard
              label="On/Off Ratio"
              value={extractedParams.onOffRatio.toFixed(2)}
            />
            <StatCard label="ΔG Range" value={extractedParams.dynamicRange.toFixed(4)} unit="µS" />
            <StatCard
              label="α_P (Potentiation)"
              value={extractedParams.alphaP.toFixed(3)}
              quality={alphaQuality(extractedParams.alphaP)}
            />
            <StatCard
              label="α_D (Depression)"
              value={extractedParams.alphaD.toFixed(3)}
              quality={alphaQuality(extractedParams.alphaD)}
            />
            <StatCard
              label="R² (P)"
              value={extractedParams.rSquaredP.toFixed(4)}
              quality={rQuality(extractedParams.rSquaredP)}
            />
            <StatCard
              label="R² (D)"
              value={extractedParams.rSquaredD.toFixed(4)}
              quality={rQuality(extractedParams.rSquaredD)}
            />
            <StatCard label="CCV" value={extractedParams.ccvPercent.toFixed(2)} unit="%" />
            <StatCard label="σ_w (Write Noise)" value={extractedParams.writeNoise.toFixed(6)} />
            <StatCard label="N Levels (P)" value={String(extractedParams.numLevelsP)} />
            <StatCard label="N Levels (D)" value={String(extractedParams.numLevelsD)} />
            <StatCard label="Memory Window" value={extractedParams.memoryWindow.toFixed(1)} unit="dB" />
            <StatCard label="Prog. Margin" value={extractedParams.programmingMargin.toFixed(1)} unit="%" />
            <StatCard
              label="Asymmetry Index"
              value={extractedParams.asymmetryIndex.toFixed(3)}
              quality={extractedParams.asymmetryIndex < 0.3 ? 'good' : extractedParams.asymmetryIndex < 0.5 ? 'ok' : 'poor'}
            />
            <StatCard
              label="Switching Uniformity"
              value={extractedParams.switchingUniformity.toFixed(3)}
              quality={extractedParams.switchingUniformity > 0.7 ? 'good' : extractedParams.switchingUniformity > 0.5 ? 'ok' : 'poor'}
            />
          </div>

          {pdChartData.length > 0 && (
            <>
              <ChartControls settings={pdChart} onChange={(s) => setPdChart((p) => ({ ...p, ...s }))} />
              <AppLineChart
                data={pdChartData}
                lines={[
                  { dataKey: 'potRaw', color: '#8494b2', name: 'P Raw', dot: true, type: 'dotted' },
                  { dataKey: 'potSmoothed', color: '#34d399', name: 'P Smoothed' },
                  { dataKey: 'potFitted', color: '#34d399', name: 'P Fitted', type: 'dashed' },
                  { dataKey: 'depRaw', color: '#8494b2', name: 'D Raw', dot: true, type: 'dotted' },
                  { dataKey: 'depSmoothed', color: '#f87171', name: 'D Smoothed' },
                  { dataKey: 'depFitted', color: '#f87171', name: 'D Fitted', type: 'dashed' },
                ]}
                xKey="index"
                xLabel={pdChart.xLabel}
                yLabel={pdChart.yLabel}
                title="Potentiation / Depression Curves"
                caption={pdChart.caption}
                plotType={pdChart.plotType}
                id="pd-curves"
              />
              {/* P/D Data Export */}
              <div className="flex gap-2 flex-wrap">
                <button
                  onClick={() => {
                    if (!extractedParams) return;
                    const pData = extractedParams.potentiationRaw.map((v: number, i: number) => ({
                      pulse: i + 1,
                      raw: v,
                      smoothed: extractedParams.potentiationSmoothed[i],
                      fitted: extractedParams.potentiationFitted[i],
                    }));
                    const dData = extractedParams.depressionRaw.map((v: number, i: number) => ({
                      pulse: i + 1,
                      raw: v,
                      smoothed: extractedParams.depressionSmoothed[i],
                      fitted: extractedParams.depressionFitted[i],
                    }));
                    exportMultiSection([
                      { label: 'Potentiation', columns: ['pulse', 'raw', 'smoothed', 'fitted'], data: pData },
                      { label: 'Depression', columns: ['pulse', 'raw', 'smoothed', 'fitted'], data: dData },
                    ], 'PD_curves', 'txt');
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-surface-alt border border-border rounded-lg text-text-muted hover:text-text hover:border-accent/50 transition-colors"
                >
                  <Download size={12} /> P/D Curves (.txt)
                </button>
                <button
                  onClick={() => {
                    if (!extractedParams) return;
                    const pData = extractedParams.potentiationRaw.map((v: number, i: number) => ({
                      pulse: i + 1,
                      raw: v,
                      smoothed: extractedParams.potentiationSmoothed[i],
                      fitted: extractedParams.potentiationFitted[i],
                    }));
                    const dData = extractedParams.depressionRaw.map((v: number, i: number) => ({
                      pulse: i + 1,
                      raw: v,
                      smoothed: extractedParams.depressionSmoothed[i],
                      fitted: extractedParams.depressionFitted[i],
                    }));
                    exportMultiSection([
                      { label: 'Potentiation', columns: ['pulse', 'raw', 'smoothed', 'fitted'], data: pData },
                      { label: 'Depression', columns: ['pulse', 'raw', 'smoothed', 'fitted'], data: dData },
                    ], 'PD_curves', 'csv');
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-surface-alt border border-border rounded-lg text-text-muted hover:text-text hover:border-accent/50 transition-colors"
                >
                  <Download size={12} /> P/D Curves (.csv)
                </button>
                <button
                  onClick={() => {
                    if (!extractedParams) return;
                    const summary = [
                      { param: 'G_min (uS)', value: extractedParams.Gmin },
                      { param: 'G_max (uS)', value: extractedParams.Gmax },
                      { param: 'On/Off Ratio', value: extractedParams.onOffRatio },
                      { param: 'Delta_G (uS)', value: extractedParams.dynamicRange },
                      { param: 'alpha_P', value: extractedParams.alphaP },
                      { param: 'alpha_D', value: extractedParams.alphaD },
                      { param: 'R2_P', value: extractedParams.rSquaredP },
                      { param: 'R2_D', value: extractedParams.rSquaredD },
                      { param: 'CCV (%)', value: extractedParams.ccvPercent },
                      { param: 'sigma_w', value: extractedParams.writeNoise },
                      { param: 'N_levels_P', value: extractedParams.numLevelsP },
                      { param: 'N_levels_D', value: extractedParams.numLevelsD },
                      { param: 'Memory_Window (dB)', value: extractedParams.memoryWindow },
                      { param: 'Prog_Margin (%)', value: extractedParams.programmingMargin },
                      { param: 'Asymmetry_Index', value: extractedParams.asymmetryIndex },
                      { param: 'Switching_Uniformity', value: extractedParams.switchingUniformity },
                    ];
                    exportChartData(summary, ['param', 'value'], 'extracted_parameters', 'txt');
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-accent/20 border border-accent/40 rounded-lg text-accent hover:bg-accent/30 transition-colors"
                >
                  <Download size={12} /> All Parameters (.txt)
                </button>
              </div>
            </>
          )}

          {deltaGData.length > 0 && (
            <>
              <ChartControls settings={dgChart} onChange={(s) => setDgChart((p) => ({ ...p, ...s }))} />
              <AppLineChart
                data={deltaGData}
                lines={[{ dataKey: 'dG', color: '#a78bfa', name: 'ΔG' }]}
                xKey="G"
                xLabel={dgChart.xLabel}
                yLabel={dgChart.yLabel}
                title="ΔG vs G — Switching Statistics"
                caption={dgChart.caption}
                plotType={dgChart.plotType}
                id="delta-g-scatter"
              />
              {/* deltaG Data Export */}
              <div className="flex gap-2">
                <button
                  onClick={() => exportChartData(deltaGData, ['G', 'dG'], 'deltaG_vs_G', 'txt')}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-surface-alt border border-border rounded-lg text-text-muted hover:text-text hover:border-accent/50 transition-colors"
                >
                  <Download size={12} /> ΔG vs G (.txt)
                </button>
                <button
                  onClick={() => exportChartData(deltaGData, ['G', 'dG'], 'deltaG_vs_G', 'csv')}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-surface-alt border border-border rounded-lg text-text-muted hover:text-text hover:border-accent/50 transition-colors"
                >
                  <Download size={12} /> ΔG vs G (.csv)
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
