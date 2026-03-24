import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import { useAppStore } from '../stores/useAppStore';
import { AppLineChart } from '../components/Chart';
import { ChartControls } from '../components/ChartControls';
import { StatCard } from '../components/StatCard';
import { ChevronDown, ChevronRight, Download } from 'lucide-react';
import { exportChartData } from '../lib/chartExport';
import type { ANNEpochResult, ANNModelType, ChartLocalSettings, FrameworkType } from '../types';

const MODEL_OPTIONS: { value: ANNModelType; label: string; desc: string }[] = [
  { value: 'perceptron', label: 'Perceptron', desc: '784 → 10 (no hidden layer)' },
  { value: 'mlp_1h', label: 'MLP (1 Hidden)', desc: '784 → H → 10' },
  { value: 'mlp_2h', label: 'MLP (2 Hidden)', desc: '784 → H1 → H2 → 10' },
  { value: 'lenet5', label: 'LeNet-5 (Python only)', desc: 'Conv→Pool→Conv→Pool→FC' },
  { value: 'cnn_simple', label: 'CNN Simple (Python only)', desc: 'Convolutional architecture' },
];

const FRAMEWORK_INFO: Record<FrameworkType, { name: string; desc: string; install: string; url: string }> = {
  pytorch: {
    name: 'Custom PyTorch',
    desc: 'Full MNIST training script with copy-and-degrade memristor simulation. Supports all architectures including CNN/LeNet-5.',
    install: 'pip install torch torchvision matplotlib numpy',
    url: 'https://pytorch.org/',
  },
  crosssim: {
    name: 'CrossSim (Sandia)',
    desc: 'Crossbar array simulator by Sandia National Labs. Models analog MAC operations with realistic device and circuit-level non-idealities.',
    install: 'git clone https://github.com/sandialabs/cross-sim.git && cd cross-sim && pip install -e .',
    url: 'https://github.com/sandialabs/cross-sim',
  },
  neurosim: {
    name: 'NeuroSim (Georgia Tech)',
    desc: 'Circuit-level benchmark framework for neuromorphic computing. Generates device configuration files and C++ parameter blocks for NeuroSim integration.',
    install: 'Clone https://github.com/neurosim/DNN_NeuroSim_V1.4',
    url: 'https://github.com/neurosim/DNN_NeuroSim_V1.4',
  },
  memtorch: {
    name: 'MemTorch',
    desc: 'PyTorch-based framework that patches standard DNN models to use memristive device models. Supports data-driven and physics-based device characterization.',
    install: 'pip install memtorch',
    url: 'https://github.com/coreylammie/MemTorch',
  },
};

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
  const [accChart, setAccChart] = useState<ChartLocalSettings>({
    xLabel: 'Epoch', yLabel: 'Accuracy (%)', plotType: 'line', caption: '',
  });
  const [expandedFramework, setExpandedFramework] = useState<string | null>(null);

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
    // LeNet-5/CNN are Python-only; fall back to MLP for in-app training
    const effectiveModelType = (annConfig.modelType === 'lenet5' || annConfig.modelType === 'cnn_simple')
      ? 'mlp_2h'
      : annConfig.modelType;

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
          gMin: extractedParams.Gmin ?? 0.1,
          gMax: extractedParams.Gmax ?? 0.3,
        },
        config: { ...annConfig, modelType: effectiveModelType },
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setIsTraining(false);
    }
  };

  const handleDownloadScript = async (framework: FrameworkType) => {
    if (!extractedParams) return;
    try {
      const scriptParams = {
        Gmin: extractedParams.Gmin,
        Gmax: extractedParams.Gmax,
        alphaP: extractedParams.alphaP,
        alphaD: extractedParams.alphaD,
        writeNoise: extractedParams.writeNoise,
        numLevelsP: extractedParams.numLevelsP,
        numLevelsD: extractedParams.numLevelsD,
      };

      let script: string;
      let defaultName: string;

      switch (framework) {
        case 'crosssim':
          script = await invoke<string>('generate_crosssim_script', { params: scriptParams });
          defaultName = 'memristor_crosssim.py';
          break;
        case 'neurosim':
          script = await invoke<string>('generate_neurosim_script', { params: scriptParams });
          defaultName = 'memristor_neurosim.py';
          break;
        case 'memtorch':
          script = await invoke<string>('generate_memtorch_script', { params: scriptParams, config: annConfig });
          defaultName = 'memristor_memtorch.py';
          break;
        default:
          script = await invoke<string>('generate_python_script', { params: scriptParams, config: annConfig });
          defaultName = 'memristor_mnist_simulation.py';
      }

      const path = await save({
        filters: [{ name: 'Python', extensions: ['py'] }],
        defaultPath: defaultName,
      });
      if (path) {
        await writeTextFile(path, script);
      }
    } catch (e) {
      if (String(e) !== 'null') {
        setError(String(e));
      }
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

  const isPythonOnly = annConfig.modelType === 'lenet5' || annConfig.modelType === 'cnn_simple';

  // Build dynamic architecture visualization
  const archLayers: { size: number | string; color: string; bgClass: string; borderClass: string }[] = [
    { size: 784, color: 'text-accent', bgClass: 'bg-accent/10', borderClass: 'border-accent/20' },
  ];
  if (isPythonOnly) {
    archLayers.push({ size: 'Conv', color: 'text-orange', bgClass: 'bg-orange/10', borderClass: 'border-orange/20' });
    archLayers.push({ size: 'Pool', color: 'text-orange', bgClass: 'bg-orange/10', borderClass: 'border-orange/20' });
    archLayers.push({ size: 120, color: 'text-purple', bgClass: 'bg-purple/10', borderClass: 'border-purple/20' });
    archLayers.push({ size: 84, color: 'text-cyan', bgClass: 'bg-cyan/10', borderClass: 'border-cyan/20' });
  } else {
    if (annConfig.modelType !== 'perceptron') {
      archLayers.push({ size: annConfig.hiddenSize, color: 'text-purple', bgClass: 'bg-purple/10', borderClass: 'border-purple/20' });
    }
    if (annConfig.modelType === 'mlp_2h') {
      archLayers.push({ size: annConfig.hiddenSize2, color: 'text-cyan', bgClass: 'bg-cyan/10', borderClass: 'border-cyan/20' });
    }
  }
  archLayers.push({ size: 10, color: 'text-green', bgClass: 'bg-green/10', borderClass: 'border-green/20' });

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
              <label className="block text-xs text-text-muted mb-1">Model Type</label>
              <select
                value={annConfig.modelType}
                onChange={(e) => setANNConfig({ modelType: e.target.value as ANNModelType })}
                className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text"
              >
                {MODEL_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label} — {o.desc}</option>
                ))}
              </select>
            </div>
            {!isPythonOnly && annConfig.modelType !== 'perceptron' && (
              <div>
                <label className="block text-xs text-text-muted mb-1">
                  Hidden Layer 1 Neurons
                </label>
                <input
                  type="number"
                  min={16}
                  max={512}
                  value={annConfig.hiddenSize}
                  onChange={(e) => setANNConfig({ hiddenSize: parseInt(e.target.value) || 128 })}
                  className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
                />
              </div>
            )}
            {!isPythonOnly && annConfig.modelType === 'mlp_2h' && (
              <div>
                <label className="block text-xs text-text-muted mb-1">
                  Hidden Layer 2 Neurons
                </label>
                <input
                  type="number"
                  min={16}
                  max={512}
                  value={annConfig.hiddenSize2}
                  onChange={(e) => setANNConfig({ hiddenSize2: parseInt(e.target.value) || 64 })}
                  className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
                />
              </div>
            )}
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
                <p>G_min = {extractedParams.Gmin.toFixed(4)} µS</p>
                <p>G_max = {extractedParams.Gmax.toFixed(4)} µS</p>
              </div>
            </div>
          )}

          {/* Dynamic architecture */}
          <div className="bg-surface rounded-xl border border-border p-4">
            <h3 className="text-sm font-medium text-text mb-2">Architecture</h3>
            <div className="flex items-center justify-center gap-2 py-3 flex-wrap">
              {archLayers.map((layer, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className={`px-3 py-2 ${layer.bgClass} border ${layer.borderClass} rounded-lg text-xs font-mono ${layer.color}`}>
                    {layer.size}
                  </div>
                  {i < archLayers.length - 1 && (
                    <span className="text-text-dim">→</span>
                  )}
                </div>
              ))}
            </div>
            {isPythonOnly && (
              <p className="text-[11px] text-amber text-center mt-1">
                CNN models run via Python script only. In-app training will use MLP (2 Hidden).
              </p>
            )}
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

          {/* Framework script generators */}
          <div className="bg-surface rounded-xl border border-border p-4 space-y-2">
            <h3 className="text-sm font-medium text-text">Export Python Scripts</h3>
            <p className="text-xs text-text-dim">Generate scripts for external frameworks with your device parameters.</p>

            {(Object.entries(FRAMEWORK_INFO) as [FrameworkType, typeof FRAMEWORK_INFO[FrameworkType]][]).map(([key, info]) => (
              <div key={key} className="border border-border rounded-lg overflow-hidden">
                <button
                  onClick={() => setExpandedFramework(expandedFramework === key ? null : key)}
                  className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-alt"
                >
                  {expandedFramework === key ? <ChevronDown size={14} className="text-text-dim" /> : <ChevronRight size={14} className="text-text-dim" />}
                  <span className="text-xs font-medium text-text flex-1">{info.name}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownloadScript(key);
                    }}
                    disabled={!extractedParams}
                    className="px-2 py-1 bg-accent/10 text-accent rounded text-[11px] font-medium hover:bg-accent/20 disabled:opacity-50"
                  >
                    Generate
                  </button>
                </button>
                {expandedFramework === key && (
                  <div className="px-3 pb-2 space-y-1">
                    <p className="text-[11px] text-text-muted">{info.desc}</p>
                    <p className="text-[11px] font-mono text-text-dim">{info.install}</p>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="p-3 bg-surface-alt rounded-lg text-xs text-text-dim leading-relaxed">
            In-app training uses synthetic MNIST data (5,000 samples) for quick parameter exploration.
            The Python scripts train on real MNIST (60,000 images) for publication results.
          </div>
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          {chartData.length > 0 ? (
            <>
              <ChartControls settings={accChart} onChange={(s) => setAccChart((p) => ({ ...p, ...s }))} />
              <AppLineChart
                data={chartData}
                lines={[
                  { dataKey: 'ideal', color: '#4f8ff7', name: 'Ideal' },
                  { dataKey: 'memristor', color: '#f87171', name: 'Memristor' },
                ]}
                xKey="epoch"
                xLabel={accChart.xLabel}
                yLabel={accChart.yLabel}
                title="Accuracy vs Epoch"
                caption={accChart.caption}
                plotType={accChart.plotType}
                heightOverride={520}
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

              {/* ANN Data Export */}
              <div className="flex gap-2 flex-wrap">
                <button
                  onClick={() => {
                    const fullData = annResults.map((r) => ({
                      epoch: r.epoch,
                      ideal_accuracy: r.idealAccuracy,
                      memristor_accuracy: r.memristorAccuracy,
                      ideal_loss: r.idealLoss,
                      memristor_loss: r.memristorLoss,
                    }));
                    exportChartData(fullData, ['epoch', 'ideal_accuracy', 'memristor_accuracy', 'ideal_loss', 'memristor_loss'], 'ANN_training_results', 'txt');
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-surface-alt border border-border rounded-lg text-text-muted hover:text-text hover:border-accent/50 transition-colors"
                >
                  <Download size={12} /> Training Data (.txt)
                </button>
                <button
                  onClick={() => {
                    const fullData = annResults.map((r) => ({
                      epoch: r.epoch,
                      ideal_accuracy: r.idealAccuracy,
                      memristor_accuracy: r.memristorAccuracy,
                      ideal_loss: r.idealLoss,
                      memristor_loss: r.memristorLoss,
                    }));
                    exportChartData(fullData, ['epoch', 'ideal_accuracy', 'memristor_accuracy', 'ideal_loss', 'memristor_loss'], 'ANN_training_results', 'csv');
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-surface-alt border border-border rounded-lg text-text-muted hover:text-text hover:border-accent/50 transition-colors"
                >
                  <Download size={12} /> Training Data (.csv)
                </button>
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
            <div className="flex items-center justify-center h-[520px] bg-surface rounded-xl border border-border text-text-dim">
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
