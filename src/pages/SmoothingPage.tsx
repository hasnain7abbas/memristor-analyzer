import { useState, useEffect, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useAppStore } from '../stores/useAppStore';
import { SmoothingControls } from '../components/SmoothingControls';
import { AppLineChart } from '../components/Chart';
import { ChartControls } from '../components/ChartControls';
import { Info } from 'lucide-react';
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

export function SmoothingPage() {
  const { uploadedTests, smoothingConfig, smoothedData, setSmoothedData } = useAppStore();
  const [selectedTest, setSelectedTest] = useState<string>('');
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [showExplanation, setShowExplanation] = useState(false);
  const [loading, setLoading] = useState(false);

  const [mainChart, setMainChart] = useState<ChartLocalSettings>({
    xLabel: 'Pulse Number', yLabel: '', plotType: 'line', caption: '',
  });
  const [residualChart, setResidualChart] = useState<ChartLocalSettings>({
    xLabel: 'Pulse Number', yLabel: 'Difference', plotType: 'line', caption: '',
  });

  const testIds = Object.keys(uploadedTests);
  const currentTest = selectedTest ? uploadedTests[selectedTest] : null;

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

  useEffect(() => {
    if (!currentTest || !selectedColumn) return;
    const col = currentTest.dataset.columns.find((c) => c.name === selectedColumn);
    if (!col || col.values.length === 0) return;

    const doSmooth = async () => {
      setLoading(true);
      try {
        const result = await invoke<number[]>('smooth_data', {
          data: col.values,
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
        setSmoothedData(`${selectedTest}_${selectedColumn}`, result);
      } catch (e) {
        console.error('Smoothing error:', e);
      } finally {
        setLoading(false);
      }
    };
    doSmooth();
  }, [currentTest, selectedColumn, smoothingConfig, selectedTest, setSmoothedData]);

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

  const rawData = currentTest?.dataset.columns.find((c) => c.name === selectedColumn)?.values || [];
  const smooth = smoothedData[`${selectedTest}_${selectedColumn}`] || [];

  const chartData = rawData.map((val, i) => ({
    index: i + 1,
    raw: val,
    smoothed: smooth[i] ?? val,
    diff: smooth[i] != null ? val - smooth[i] : 0,
  }));

  const r2 = useMemo(() => computeR2(rawData, smooth), [rawData, smooth]);
  const shapeScore = useMemo(() => computeShapePreservation(rawData, smooth), [rawData, smooth]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text mb-1">Data Smoothing</h2>
        <p className="text-sm text-text-muted">
          Apply smoothing algorithms to your data before parameter extraction.
        </p>
      </div>

      {/* Dataset + column selection */}
      <div className="flex gap-4">
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

      <SmoothingControls />

      {loading && (
        <div className="flex items-center gap-2 text-sm text-text-muted">
          <div className="w-4 h-4 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
          Smoothing...
        </div>
      )}

      {/* Quality metrics */}
      {smooth.length > 0 && smoothingConfig.method !== 'none' && (
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
            lines={[
              { dataKey: 'raw', color: '#8494b2', name: 'Raw', dot: true, type: 'dotted' },
              { dataKey: 'smoothed', color: '#4f8ff7', name: 'Smoothed' },
            ]}
            xKey="index"
            xLabel={mainChart.xLabel}
            yLabel={mainChart.yLabel}
            title="Raw vs Smoothed"
            caption={mainChart.caption}
            plotType={mainChart.plotType}
            id="smoothing-comparison"
          />

          {smoothingConfig.method !== 'none' && (
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
