import { useState, useEffect, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useAppStore } from '../stores/useAppStore';
import { SmoothingControls } from '../components/SmoothingControls';
import { AppLineChart } from '../components/Chart';
import { ChartControls } from '../components/ChartControls';
import { Info, Download } from 'lucide-react';
import { exportChartData } from '../lib/chartExport';
import type { ChartLocalSettings } from '../types';

function computeR2(raw: number[], smoothed: number[]): number {
  if (raw.length < 2 || raw.length !== smoothed.length) return 0;
  const mean = raw.reduce((a, b) => a + b, 0) / raw.length;
  const ssTot = raw.reduce((a, v) => a + (v - mean) ** 2, 0);
  const ssRes = raw.reduce((a, v, i) => a + (v - smoothed[i]) ** 2, 0);
  return ssTot > 1e-15 ? 1 - ssRes / ssTot : 0;
}

function computeShapePreservation(raw: number[], smoothed: number[]): number {
  // Measures how well the smoothed curve preserves the derivative sign pattern
  if (raw.length < 3) return 1;
  let matchCount = 0;
  let total = 0;
  for (let i = 1; i < raw.length; i++) {
    const rawSign = Math.sign(raw[i] - raw[i - 1]);
    const smSign = Math.sign(smoothed[i] - smoothed[i - 1]);
    if (rawSign === 0 || smSign === 0) continue;
    total++;
    if (rawSign === smSign) matchCount++;
  }
  return total > 0 ? matchCount / total : 1;
}

/** Segment flat data into individual P/D cycle blocks. */
function segmentIntoCycles(values: number[], pulsesPerP: number, pulsesPerD: number): { pCycles: number[][]; dCycles: number[][] } {
  const cycleLen = pulsesPerP + pulsesPerD;
  const numCycles = Math.floor(values.length / cycleLen);
  const pCycles: number[][] = [];
  const dCycles: number[][] = [];
  for (let c = 0; c < numCycles; c++) {
    const start = c * cycleLen;
    pCycles.push(values.slice(start, start + pulsesPerP));
    dCycles.push(values.slice(start + pulsesPerP, start + cycleLen));
  }
  return { pCycles, dCycles };
}

/** Average multiple arrays of same length into one. */
function averageArrays(arrays: number[][]): number[] {
  if (arrays.length === 0) return [];
  if (arrays.length === 1) return arrays[0];
  const len = Math.min(...arrays.map(a => a.length));
  const result: number[] = new Array(len).fill(0);
  for (const arr of arrays) {
    for (let i = 0; i < len; i++) result[i] += arr[i];
  }
  for (let i = 0; i < len; i++) result[i] /= arrays.length;
  return result;
}

type ViewMode = 'raw' | 'per_cycle' | 'avg_cycle';

export function SmoothingPage() {
  const { uploadedTests, smoothingConfig, smoothedData, setSmoothedData, cycleConfig, setCycleConfig } = useAppStore();
  const [selectedTest, setSelectedTest] = useState<string>('');
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [showExplanation, setShowExplanation] = useState(false);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('avg_cycle');

  const [mainChart, setMainChart] = useState<ChartLocalSettings>({
    xLabel: 'Pulse Number', yLabel: '', plotType: 'line', caption: '',
  });
  const [residualChart, setResidualChart] = useState<ChartLocalSettings>({
    xLabel: 'Pulse Number', yLabel: 'Difference', plotType: 'line', caption: '',
  });

  const testIds = Object.keys(uploadedTests);
  const currentTest = selectedTest ? uploadedTests[selectedTest] : null;

  // Determine if this is a P/D test that needs cycle segmentation
  const isPDTest = selectedTest === 'pd_training' || selectedTest === 'ltp_ltd';

  useEffect(() => {
    if (testIds.length > 0 && !selectedTest) {
      setSelectedTest(testIds[0]);
    }
  }, [testIds, selectedTest]);

  useEffect(() => {
    if (currentTest) {
      const numCols = currentTest.dataset.columns.filter((c) => c.values.length > 0);
      if (numCols.length > 0 && !selectedColumn) {
        const condCol = numCols.find((c) => /conductance|cond|g_?us|^g$/i.test(c.name));
        const col = condCol?.name || numCols[0].name;
        setSelectedColumn(col);
        setMainChart((s) => ({ ...s, yLabel: col }));
      }
    }
  }, [currentTest, selectedColumn]);

  // Get raw values for the selected column
  const rawValues = useMemo(() => {
    if (!currentTest || !selectedColumn) return [];
    const col = currentTest.dataset.columns.find((c) => c.name === selectedColumn);
    return col?.values ?? [];
  }, [currentTest, selectedColumn]);

  // Segment into cycles if this is P/D data
  const cycleData = useMemo(() => {
    if (!isPDTest || rawValues.length === 0) return null;
    return segmentIntoCycles(rawValues, cycleConfig.pulsesPerP, cycleConfig.pulsesPerD);
  }, [isPDTest, rawValues, cycleConfig.pulsesPerP, cycleConfig.pulsesPerD]);

  // Compute the data to smooth based on view mode
  const dataToSmooth = useMemo(() => {
    if (!isPDTest || !cycleData || viewMode === 'raw') return rawValues;
    if (viewMode === 'avg_cycle') {
      // Average P cycles and D cycles separately, then concatenate
      const pAvg = averageArrays(cycleData.pCycles);
      const dAvg = averageArrays(cycleData.dCycles);
      return [...pAvg, ...dAvg];
    }
    return rawValues;
  }, [isPDTest, cycleData, viewMode, rawValues]);

  useEffect(() => {
    if (dataToSmooth.length === 0) return;

    const doSmooth = async () => {
      setLoading(true);
      try {
        const result = await invoke<number[]>('smooth_data', {
          data: dataToSmooth,
          method: smoothingConfig.method,
          windowSize: smoothingConfig.windowSize,
          polyOrder: smoothingConfig.method === 'savitzky_golay' ? smoothingConfig.polyOrder : null,
          removeOutliers: smoothingConfig.removeOutliers,
          outlierSigma: smoothingConfig.outlierSigma,
          strength: smoothingConfig.strength,
          enforceMonotonic: smoothingConfig.enforceMonotonic,
          monotonicDirection: smoothingConfig.monotonicDirection,
          bandwidth: smoothingConfig.bandwidth,
        });
        setSmoothedData(`${selectedTest}_${selectedColumn}_${viewMode}`, result);
      } catch (e) {
        console.error('Smoothing error:', e);
      } finally {
        setLoading(false);
      }
    };
    doSmooth();
  }, [dataToSmooth, smoothingConfig, selectedTest, selectedColumn, viewMode, setSmoothedData]);

  if (testIds.length === 0) {
    return (
      <div className="text-center py-20 text-text-muted space-y-2">
        <p className="text-lg font-medium">No data uploaded yet</p>
        <p className="text-sm">
          Go to the <span className="text-accent font-medium">Upload</span> tab to load your measurement data,
          or use the <span className="text-purple font-medium">Load Demo Data</span> button to test the pipeline.
        </p>
      </div>
    );
  }

  const smooth = smoothedData[`${selectedTest}_${selectedColumn}_${viewMode}`] || [];

  // Build chart data based on view mode
  let chartData: any[] = [];
  let chartLines: any[] = [];

  if (viewMode === 'per_cycle' && cycleData) {
    // Show all cycles overlaid, each P cycle as a separate line
    const maxCyclesToShow = Math.min(cycleData.pCycles.length, 10);
    const pLen = cycleData.pCycles[0]?.length ?? 0;
    const dLen = cycleData.dCycles[0]?.length ?? 0;
    const totalLen = pLen + dLen;

    const colors = ['#4f8ff7', '#f87171', '#34d399', '#fbbf24', '#a78bfa', '#22d3ee', '#f472b6', '#fb923c', '#2dd4bf', '#818cf8'];

    for (let i = 0; i < totalLen; i++) {
      const point: any = { index: i + 1 };
      for (let c = 0; c < maxCyclesToShow; c++) {
        if (i < pLen) {
          point[`cycle_${c}`] = cycleData.pCycles[c]?.[i];
        } else {
          point[`cycle_${c}`] = cycleData.dCycles[c]?.[i - pLen];
        }
      }
      chartData.push(point);
    }

    for (let c = 0; c < maxCyclesToShow; c++) {
      chartLines.push({
        dataKey: `cycle_${c}`,
        color: colors[c % colors.length],
        name: `Cycle ${c + 1}`,
        dot: false,
      });
    }
  } else {
    // Raw or avg_cycle mode: show raw vs smoothed
    chartData = dataToSmooth.map((val, i) => ({
      index: i + 1,
      raw: val,
      smoothed: smooth[i] ?? val,
      diff: smooth[i] != null ? val - smooth[i] : 0,
    }));
    chartLines = [
      { dataKey: 'raw', color: '#8494b2', name: 'Raw', dot: true, type: 'dotted' as const },
      { dataKey: 'smoothed', color: '#4f8ff7', name: 'Smoothed' },
    ];
  }

  const r2 = useMemo(() => computeR2(dataToSmooth, smooth), [dataToSmooth, smooth]);
  const shapeScore = useMemo(() => computeShapePreservation(dataToSmooth, smooth), [dataToSmooth, smooth]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text mb-1">Data Smoothing</h2>
        <p className="text-sm text-text-muted">
          Apply smoothing algorithms to your data before parameter extraction.
        </p>
      </div>

      {/* Dataset + column selection */}
      <div className="flex gap-4 flex-wrap">
        <div>
          <label className="block text-xs text-text-muted mb-1">Dataset</label>
          <select
            value={selectedTest}
            onChange={(e) => {
              setSelectedTest(e.target.value);
              setSelectedColumn('');
            }}
            className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text"
          >
            {testIds.map((id) => (
              <option key={id} value={id}>
                {uploadedTests[id].dataset.filename}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Column</label>
          <select
            value={selectedColumn}
            onChange={(e) => {
              setSelectedColumn(e.target.value);
              setMainChart((s) => ({ ...s, yLabel: e.target.value }));
            }}
            className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text"
          >
            {currentTest?.dataset.columns
              .filter((c) => c.values.length > 0)
              .map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
          </select>
        </div>
      </div>

      {/* Cycle structure controls — shown for P/D tests */}
      {isPDTest && (
        <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
          <h3 className="text-sm font-medium text-text">Cycle Structure</h3>
          <p className="text-xs text-text-muted">
            Your data contains repeating P/D cycles (e.g., 50 positive pulses then 50 negative pulses, repeated N times).
            Specify the cycle structure so smoothing and visualization work correctly.
          </p>
          <div className="flex items-end gap-4 flex-wrap">
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
            {rawValues.length > 0 && (
              <span className="text-xs text-text-dim">
                {Math.floor(rawValues.length / (cycleConfig.pulsesPerP + cycleConfig.pulsesPerD))} cycle(s) from {rawValues.length} data points
              </span>
            )}
          </div>

          {/* View mode toggle */}
          <div className="flex gap-2 pt-1">
            {(['raw', 'per_cycle', 'avg_cycle'] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1.5 text-xs rounded-lg border ${
                  viewMode === mode
                    ? 'bg-accent/20 border-accent text-accent'
                    : 'bg-surface-alt border-border text-text-muted hover:text-text'
                }`}
              >
                {mode === 'raw' ? 'Raw Sequence' : mode === 'per_cycle' ? 'Per-Cycle Overlay' : 'Average Cycle'}
              </button>
            ))}
          </div>
        </div>
      )}

      <SmoothingControls />

      {loading && (
        <div className="flex items-center gap-2 text-sm text-text-muted">
          <div className="w-4 h-4 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
          Smoothing...
        </div>
      )}

      {/* Quality metrics */}
      {smooth.length > 0 && smoothingConfig.method !== 'none' && viewMode !== 'per_cycle' && (
        <div className="flex gap-4">
          <div className="px-3 py-2 bg-surface rounded-lg border border-border">
            <span className="text-xs text-text-muted">R² (fit quality): </span>
            <span className={`text-sm font-mono font-medium ${r2 >= 0.95 ? 'text-green' : r2 >= 0.8 ? 'text-amber' : 'text-red'}`}>
              {r2.toFixed(4)}
            </span>
          </div>
          <div className="px-3 py-2 bg-surface rounded-lg border border-border">
            <span className="text-xs text-text-muted">Shape preservation: </span>
            <span className={`text-sm font-mono font-medium ${shapeScore >= 0.8 ? 'text-green' : shapeScore >= 0.6 ? 'text-amber' : 'text-red'}`}>
              {(shapeScore * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {chartData.length > 0 && (
        <>
          <ChartControls settings={mainChart} onChange={(s) => setMainChart((p) => ({ ...p, ...s }))} />
          <AppLineChart
            data={chartData}
            lines={chartLines}
            xKey="index"
            xLabel={mainChart.xLabel}
            yLabel={mainChart.yLabel}
            title={viewMode === 'per_cycle' ? 'Per-Cycle Overlay' : viewMode === 'avg_cycle' ? 'Average Cycle (Raw vs Smoothed)' : 'Raw vs Smoothed'}
            caption={mainChart.caption}
            plotType={mainChart.plotType}
            id="smoothing-comparison"
          />

          {/* Data Export */}
          <div className="flex gap-2">
            <button
              onClick={() => {
                const cols = Object.keys(chartData[0] || {});
                exportChartData(chartData, cols, `smoothing_${viewMode}`, 'txt');
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-surface-alt border border-border rounded-lg text-text-muted hover:text-text hover:border-accent/50 transition-colors"
            >
              <Download size={12} /> Export .txt (Tab-delimited)
            </button>
            <button
              onClick={() => {
                const cols = Object.keys(chartData[0] || {});
                exportChartData(chartData, cols, `smoothing_${viewMode}`, 'csv');
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-surface-alt border border-border rounded-lg text-text-muted hover:text-text hover:border-accent/50 transition-colors"
            >
              <Download size={12} /> Export .csv
            </button>
          </div>

          {smoothingConfig.method !== 'none' && viewMode !== 'per_cycle' && (
            <>
              <ChartControls settings={residualChart} onChange={(s) => setResidualChart((p) => ({ ...p, ...s }))} />
              <AppLineChart
                data={chartData}
                lines={[{ dataKey: 'diff', color: '#a78bfa', name: 'Difference (Raw - Smoothed)' }]}
                xKey="index"
                xLabel={residualChart.xLabel}
                yLabel={residualChart.yLabel}
                title="Residuals"
                caption={residualChart.caption}
                plotType={residualChart.plotType}
                heightOverride={200}
                id="smoothing-residuals"
              />
            </>
          )}
        </>
      )}

      {/* Explanation panel */}
      <div className="border border-border rounded-xl">
        <button
          onClick={() => setShowExplanation(!showExplanation)}
          className="w-full flex items-center gap-2 px-4 py-3 text-left"
        >
          <Info size={16} className="text-accent" />
          <span className="text-sm font-medium text-text">Smoothing Methods Explained</span>
        </button>
        {showExplanation && (
          <div className="px-4 pb-4 space-y-3 text-sm text-text-muted">
            <div>
              <p className="font-medium text-cyan">Savitzky-Golay (Recommended for general use)</p>
              <p>
                Fits local polynomials to preserve peak shapes while removing noise. Window 5-7 with
                order 2 works for most memristor data. Uses reflected boundary padding to eliminate edge discontinuities.
              </p>
            </div>
            <div>
              <p className="font-medium text-cyan">LOESS/LOWESS (Best for non-linear patterns)</p>
              <p>
                Locally weighted scatterplot smoothing. Uses tricube weights and local linear regression.
                Bandwidth controls the fraction of data used for each local fit (0.1 = more detail, 0.5 = smoother).
                Best for preserving sigmoid/saturation patterns in P/D curves.
              </p>
            </div>
            <div>
              <p className="font-medium text-cyan">Gaussian Kernel</p>
              <p>
                Gaussian-weighted moving average. Preserves shape better than simple moving average because
                center points are weighted more heavily. Window size controls the spread of the Gaussian.
              </p>
            </div>
            <div>
              <p className="font-medium text-cyan">Moving Average</p>
              <p>
                Simple mean of neighboring points. Fast but can flatten sharp transitions.
              </p>
            </div>
            <div>
              <p className="font-medium text-cyan">Median Filter</p>
              <p>
                Replaces each point with the median of its neighbors. Best for removing single-point
                spike artifacts.
              </p>
            </div>
            <div>
              <p className="font-medium text-cyan">Monotonicity Enforcement</p>
              <p>
                Uses isotonic regression (pool adjacent violators algorithm) to enforce that the smoothed
                curve is monotonically increasing or decreasing. Critical for memristor P/D data where
                potentiation should always increase and depression should always decrease.
              </p>
            </div>
            <div>
              <p className="font-medium text-cyan">Strength Blending</p>
              <p>
                Controls how much smoothing is applied: 0.0 = raw data, 1.0 = fully smoothed.
                Values in between linearly blend raw and smoothed data.
              </p>
            </div>
            <div>
              <p className="font-medium text-cyan">Outlier Removal</p>
              <p>
                Points more than Nσ from the mean are replaced with the average of their immediate
                neighbors. Applied BEFORE smoothing.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
