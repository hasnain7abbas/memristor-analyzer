import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useAppStore } from '../stores/useAppStore';
import { FormulaCard } from '../components/FormulaCard';
import { StatCard } from '../components/StatCard';
import { AppLineChart } from '../components/Chart';
import { ChartControls } from '../components/ChartControls';
import type { ChartLocalSettings } from '../types';

const FORMULAS = [
  {
    title: 'Step 1: Current → Conductance',
    formula: 'G = I_read / V_read\n\nG (µS) = I (µA) / V (V)',
    explanation:
      'If your Keithley records current (µA) at a fixed read voltage, divide to get conductance in µS (microsiemens).',
    example:
      'I_read = 50 µA, V_read = 0.1 V\nG = 50 / 0.1 = 500 µS',
  },
  {
    title: 'Step 2: G_min, G_max, On/Off Ratio',
    formula:
      'G_min = min(potentiation curve)\nG_max = max(potentiation curve)\nOn/Off = G_max / G_min\nΔG = G_max - G_min',
    explanation:
      'The conductance window between LRS and HRS. Larger windows give more analog levels for the ANN.',
    example: 'G_min = 10 µS, G_max = 100 µS\nOn/Off = 10.0, ΔG = 90 µS',
  },
  {
    title: 'Step 3: Non-Linearity α — THE KEY PARAMETER',
    formula:
      'G(n) = G_start + (G_end - G_start) × [1 - exp(-α·n/N)] / [1 - exp(-α)]\n\nα → 0: perfectly linear\nα ≈ 1-2: mild non-linearity\nα > 4: severe (hurts ANN performance)',
    explanation:
      'α quantifies how the conductance update "saturates" as the device approaches its limit. Lower α means more uniform weight updates, which is better for training ANNs.',
    example:
      'If α_P = 1.5 and α_D = 2.0:\nMild non-linearity. Expect ~5-15% accuracy drop vs ideal.',
    highlight: true,
  },
  {
    title: 'Step 4: CCV & Write Noise',
    formula:
      'ΔG[i] = G[i+1] - G[i]\nCCV% = std(|ΔG|) / mean(|ΔG|) × 100%\nσ_w = std(ΔG) / (G_max - G_min)',
    explanation:
      'Cycle-to-cycle variation (CCV) measures how reproducible each weight update is. Write noise σ_w is the normalized standard deviation of conductance changes.',
    example:
      'mean(|ΔG|) = 2.0 µS, std(|ΔG|) = 0.5 µS\nCCV = 0.5/2.0 × 100 = 25%\nσ_w = 0.5/90 = 0.0056',
  },
  {
    title: 'Step 5: ΔG vs G Scatter Plot',
    formula: 'For each pulse i:\n  x = G[i]\n  y = G[i+1] - G[i]\n\nPlot (x, y) for all potentiation and depression pulses.',
    explanation:
      'This scatter plot reveals the state-dependent switching behavior. Ideally, ΔG should be constant (horizontal line). A slope indicates non-linearity.',
  },
  {
    title: 'Step 6: PPF Index',
    formula: 'PPF Index = (A2 - A1) / A1 × 100%\n\nDouble exponential fit:\nPPF(Δt) = C₁·exp(-Δt/τ₁) + C₂·exp(-Δt/τ₂)',
    explanation:
      'Paired-pulse facilitation measures short-term synaptic plasticity. τ₁ and τ₂ correspond to fast and slow decay timescales.',
  },
];

export function ParametersPage() {
  const { uploadedTests, smoothingConfig, extractedParams, setExtractedParams, vRead, setVRead, isCurrentInput } =
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

      if (typeColName) {
        const typeVals = ds.rows.map((r) => String(r[typeColName] || '').toLowerCase());
        pRaw = values.filter((_, i) => /^p|pot|ltp/i.test(typeVals[i]));
        dRaw = values.filter((_, i) => /^d|dep|ltd/i.test(typeVals[i]));
        if (pRaw.length === 0 || dRaw.length === 0) {
          const peakIdx = values.indexOf(Math.max(...values));
          pRaw = values.slice(0, peakIdx + 1);
          dRaw = values.slice(peakIdx);
        }
      } else {
        const peakIdx = values.indexOf(Math.max(...values));
        if (peakIdx > values.length * 0.8 || peakIdx < values.length * 0.2) {
          const mid = Math.floor(values.length / 2);
          pRaw = values.slice(0, mid);
          dRaw = values.slice(mid);
        } else {
          pRaw = values.slice(0, peakIdx + 1);
          dRaw = values.slice(peakIdx);
        }
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

      setExtractedParams(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const rQuality = (r2: number) => (r2 >= 0.95 ? 'good' : r2 >= 0.85 ? 'ok' : 'poor');

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
        <div className="flex items-end gap-4">
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
              quality={extractedParams.alphaP < 2 ? 'good' : extractedParams.alphaP < 4 ? 'ok' : 'poor'}
            />
            <StatCard
              label="α_D (Depression)"
              value={extractedParams.alphaD.toFixed(3)}
              quality={extractedParams.alphaD < 2 ? 'good' : extractedParams.alphaD < 4 ? 'ok' : 'poor'}
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
            </>
          )}
        </div>
      )}
    </div>
  );
}
