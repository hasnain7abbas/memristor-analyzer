import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import { useAppStore } from '../stores/useAppStore';
import { AppLineChart } from '../components/Chart';
import { StatCard } from '../components/StatCard';
import type { ANNEpochResult } from '../types';

export function ANNPage() {
  const {
    extractedParams,
    annConfig,
    setANNConfig,
    annResults,
    setANNResults,
    addANNResult,
    isTraining,
    setIsTraining,
  } = useAppStore();
  const [error, setError] = useState<string | null>(null);

  // Listen for progress events
  useEffect(() => {
    const unlisten = listen<ANNEpochResult>('ann-progress', (event) => {
      addANNResult(event.payload);
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [addANNResult]);

  const handleTrain = async () => {
    if (!extractedParams) return;
    setIsTraining(true);
    setANNResults([]);
    setError(null);

    try {
      await invoke('train_ann', {
        params: {
          alphaP: extractedParams.alphaP,
          alphaD: extractedParams.alphaD,
          writeNoise: extractedParams.writeNoise,
          numLevelsP: extractedParams.numLevelsP,
          numLevelsD: extractedParams.numLevelsD,
        },
        config: annConfig,
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setIsTraining(false);
    }
  };

  const handleDownloadScript = async () => {
    if (!extractedParams) return;
    try {
      const script = await invoke<string>('generate_python_script', {
        params: {
          Gmin: extractedParams.Gmin,
          Gmax: extractedParams.Gmax,
          alphaP: extractedParams.alphaP,
          alphaD: extractedParams.alphaD,
          writeNoise: extractedParams.writeNoise,
          numLevelsP: extractedParams.numLevelsP,
          numLevelsD: extractedParams.numLevelsD,
        },
        config: annConfig,
      });

      const path = await save({
        filters: [{ name: 'Python', extensions: ['py'] }],
        defaultPath: 'memristor_mnist_simulation.py',
      });
      if (path) {
        await writeTextFile(path, script);
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const chartData = annResults.map((r) => ({
    epoch: r.epoch,
    ideal: r.idealAccuracy,
    memristor: r.memristorAccuracy,
  }));

  const bestIdeal = annResults.length > 0 ? Math.max(...annResults.map((r) => r.idealAccuracy)) : 0;
  const bestMem = annResults.length > 0 ? Math.max(...annResults.map((r) => r.memristorAccuracy)) : 0;
  const drop = bestIdeal - bestMem;
  const progress = isTraining && annResults.length > 0
    ? (annResults.length / annConfig.epochs) * 100
    : 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text mb-1">ANN / MNIST Simulation</h2>
        <p className="text-sm text-text-muted">
          Train a neural network using your device parameters to evaluate memristor suitability for neuromorphic computing.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
        {/* Left panel */}
        <div className="space-y-4">
          {/* Config */}
          <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
            <h3 className="text-sm font-medium text-text">Network Configuration</h3>
            <div>
              <label className="block text-xs text-text-muted mb-1">Hidden Neurons</label>
              <input
                type="number"
                min={16}
                max={512}
                value={annConfig.hiddenSize}
                onChange={(e) => setANNConfig({ hiddenSize: parseInt(e.target.value) || 128 })}
                className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
              />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">Epochs</label>
              <input
                type="number"
                min={5}
                max={200}
                value={annConfig.epochs}
                onChange={(e) => setANNConfig({ epochs: parseInt(e.target.value) || 50 })}
                className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
              />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">Learning Rate</label>
              <input
                type="number"
                step={0.001}
                min={0.001}
                max={0.5}
                value={annConfig.learningRate}
                onChange={(e) => setANNConfig({ learningRate: parseFloat(e.target.value) || 0.03 })}
                className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
              />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">Batch Size</label>
              <input
                type="number"
                min={1}
                max={128}
                value={annConfig.batchSize}
                onChange={(e) => setANNConfig({ batchSize: parseInt(e.target.value) || 32 })}
                className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
              />
            </div>
          </div>

          {/* Device params summary */}
          {extractedParams && (
            <div className="bg-surface rounded-xl border border-border p-4 space-y-2">
              <h3 className="text-sm font-medium text-text">Device Parameters</h3>
              <div className="text-xs space-y-1 font-mono text-text-muted">
                <p>α_P = {extractedParams.alphaP.toFixed(3)}</p>
                <p>α_D = {extractedParams.alphaD.toFixed(3)}</p>
                <p>σ_w = {extractedParams.writeNoise.toFixed(6)}</p>
                <p>N_P = {extractedParams.numLevelsP}</p>
                <p>N_D = {extractedParams.numLevelsD}</p>
              </div>
            </div>
          )}

          {/* Architecture */}
          <div className="bg-surface rounded-xl border border-border p-4">
            <h3 className="text-sm font-medium text-text mb-2">Architecture</h3>
            <div className="flex items-center justify-center gap-2 py-3">
              <div className="px-3 py-2 bg-accent/10 border border-accent/20 rounded-lg text-xs font-mono text-accent">
                784
              </div>
              <span className="text-text-dim">→</span>
              <div className="px-3 py-2 bg-purple/10 border border-purple/20 rounded-lg text-xs font-mono text-purple">
                {annConfig.hiddenSize}
              </div>
              <span className="text-text-dim">→</span>
              <div className="px-3 py-2 bg-green/10 border border-green/20 rounded-lg text-xs font-mono text-green">
                10
              </div>
            </div>
          </div>

          {/* Buttons */}
          <button
            onClick={handleTrain}
            disabled={isTraining || !extractedParams}
            className="w-full px-4 py-3 bg-accent text-white rounded-xl text-sm font-medium hover:bg-accent/80 disabled:opacity-50"
          >
            {isTraining ? `Training... ${Math.round(progress)}%` : 'Train In-App'}
          </button>

          {isTraining && (
            <div className="w-full bg-surface-alt rounded-full h-2">
              <div
                className="bg-accent h-2 rounded-full transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}

          <button
            onClick={handleDownloadScript}
            disabled={!extractedParams}
            className="w-full px-4 py-2.5 bg-surface border border-border text-text rounded-xl text-sm font-medium hover:bg-surface-alt disabled:opacity-50"
          >
            Download Python Script
          </button>

          <div className="p-3 bg-surface-alt rounded-lg text-xs text-text-dim leading-relaxed">
            In-app training uses synthetic MNIST data (3000 samples) for quick parameter exploration.
            The Python script trains on real MNIST (60,000 images) for publication results.
          </div>
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          {chartData.length > 0 ? (
            <>
              <AppLineChart
                data={chartData}
                lines={[
                  { dataKey: 'ideal', color: '#4f8ff7', name: 'Ideal' },
                  { dataKey: 'memristor', color: '#f87171', name: 'Memristor' },
                ]}
                xKey="epoch"
                xLabel="Epoch"
                yLabel="Accuracy (%)"
                title="Accuracy vs Epoch"
                style={{ height: 400 }}
                id="ann-accuracy"
              />

              <div className="grid grid-cols-3 gap-3">
                <StatCard label="Best Ideal" value={`${bestIdeal.toFixed(1)}%`} color="text-accent" />
                <StatCard label="Best Memristor" value={`${bestMem.toFixed(1)}%`} color="text-red" />
                <StatCard
                  label="Accuracy Drop"
                  value={`${drop.toFixed(1)}%`}
                  quality={drop < 10 ? 'good' : drop < 25 ? 'ok' : 'poor'}
                />
              </div>

              <div className="bg-surface rounded-xl border border-border p-4 text-sm text-text-muted space-y-2">
                <p className="font-medium text-text">What do the curves mean?</p>
                <p>
                  <span className="text-accent font-medium">Blue (Ideal)</span>: Standard ANN with perfect weight updates — your upper bound.
                </p>
                <p>
                  <span className="text-red font-medium">Red (Memristor)</span>: ANN with your device's non-linearity (α), quantization (N levels), and write noise (σ_w).
                </p>
                <p>
                  The gap between the curves shows how much your device's non-idealities degrade ANN performance.
                  Smaller gaps indicate better memristor suitability for neuromorphic applications.
                </p>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-[400px] bg-surface rounded-xl border border-border text-text-dim">
              {extractedParams
                ? 'Click "Train In-App" to start the simulation'
                : 'Extract parameters first on the Parameters tab'}
            </div>
          )}
        </div>
      </div>

      {error && <p className="text-sm text-red">{error}</p>}
    </div>
  );
}
