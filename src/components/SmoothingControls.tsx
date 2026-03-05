import { useAppStore } from '../stores/useAppStore';

export function SmoothingControls() {
  const { smoothingConfig, setSmoothingConfig } = useAppStore();

  return (
    <div className="flex flex-wrap items-end gap-4 p-4 bg-surface rounded-xl border border-border">
      <div>
        <label className="block text-xs text-text-muted mb-1">Method</label>
        <select
          value={smoothingConfig.method}
          onChange={(e) => setSmoothingConfig({ method: e.target.value as any })}
          className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text min-w-[180px]"
        >
          <option value="none">None</option>
          <option value="moving_avg">Moving Average</option>
          <option value="median">Median Filter</option>
          <option value="savitzky_golay">Savitzky-Golay</option>
          <option value="loess">LOESS/LOWESS</option>
          <option value="gaussian">Gaussian Kernel</option>
        </select>
      </div>

      {smoothingConfig.method !== 'none' && smoothingConfig.method !== 'loess' && (
        <div>
          <label className="block text-xs text-text-muted mb-1">Window Size</label>
          <input
            type="number"
            min={3}
            max={21}
            step={2}
            value={smoothingConfig.windowSize}
            onChange={(e) => {
              let v = parseInt(e.target.value);
              if (v % 2 === 0) v += 1;
              setSmoothingConfig({ windowSize: Math.max(3, Math.min(21, v)) });
            }}
            className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text w-20 font-mono"
          />
        </div>
      )}

      {smoothingConfig.method === 'savitzky_golay' && (
        <div>
          <label className="block text-xs text-text-muted mb-1">Poly Order</label>
          <input
            type="number"
            min={1}
            max={5}
            value={smoothingConfig.polyOrder}
            onChange={(e) =>
              setSmoothingConfig({
                polyOrder: Math.max(1, Math.min(5, parseInt(e.target.value) || 2)),
              })
            }
            className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text w-20 font-mono"
          />
        </div>
      )}

      {smoothingConfig.method === 'loess' && (
        <div>
          <label className="block text-xs text-text-muted mb-1">Bandwidth (0.05-0.9)</label>
          <input
            type="number"
            min={0.05}
            max={0.9}
            step={0.05}
            value={smoothingConfig.bandwidth}
            onChange={(e) =>
              setSmoothingConfig({ bandwidth: parseFloat(e.target.value) || 0.3 })
            }
            className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text w-24 font-mono"
          />
        </div>
      )}

      {smoothingConfig.method !== 'none' && (
        <div>
          <label className="block text-xs text-text-muted mb-1">
            Strength: {smoothingConfig.strength.toFixed(2)}
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={smoothingConfig.strength}
            onChange={(e) => setSmoothingConfig({ strength: parseFloat(e.target.value) })}
            className="w-28 accent-accent"
          />
        </div>
      )}

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={smoothingConfig.removeOutliers}
          onChange={(e) => setSmoothingConfig({ removeOutliers: e.target.checked })}
          className="w-4 h-4 rounded border-border accent-accent"
          id="outlier-check"
        />
        <label htmlFor="outlier-check" className="text-sm text-text-muted">
          Remove Outliers
        </label>
      </div>

      {smoothingConfig.removeOutliers && (
        <div>
          <label className="block text-xs text-text-muted mb-1">Outlier σ</label>
          <input
            type="number"
            min={1.0}
            max={5.0}
            step={0.1}
            value={smoothingConfig.outlierSigma}
            onChange={(e) =>
              setSmoothingConfig({ outlierSigma: parseFloat(e.target.value) || 2.5 })
            }
            className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text w-20 font-mono"
          />
        </div>
      )}

      {smoothingConfig.method !== 'none' && (
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={smoothingConfig.enforceMonotonic}
            onChange={(e) => setSmoothingConfig({ enforceMonotonic: e.target.checked })}
            className="w-4 h-4 rounded border-border accent-accent"
            id="mono-check"
          />
          <label htmlFor="mono-check" className="text-sm text-text-muted">
            Enforce Monotonicity
          </label>
        </div>
      )}

      {smoothingConfig.enforceMonotonic && smoothingConfig.method !== 'none' && (
        <div>
          <label className="block text-xs text-text-muted mb-1">Direction</label>
          <select
            value={smoothingConfig.monotonicDirection}
            onChange={(e) => setSmoothingConfig({ monotonicDirection: e.target.value as any })}
            className="bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text"
          >
            <option value="auto">Auto-detect</option>
            <option value="increasing">Increasing (P curve)</option>
            <option value="decreasing">Decreasing (D curve)</option>
          </select>
        </div>
      )}
    </div>
  );
}
