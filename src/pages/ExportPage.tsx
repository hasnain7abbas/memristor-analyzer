import { useAppStore } from '../stores/useAppStore';

const PRESETS = [
  { name: 'Frontiers/Elsevier (single col)', width: 600, height: 450, fontFamily: 'Times New Roman', axisFontSize: 12, titleFontSize: 14, tickFontSize: 11, dpi: 300 },
  { name: 'Frontiers/Elsevier (double col)', width: 1200, height: 450, fontFamily: 'Times New Roman', axisFontSize: 14, titleFontSize: 16, tickFontSize: 12, dpi: 300 },
  { name: 'Nature/Science', width: 600, height: 500, fontFamily: 'Helvetica', axisFontSize: 11, titleFontSize: 13, tickFontSize: 10, dpi: 600 },
  { name: 'ACS Journals', width: 600, height: 450, fontFamily: 'Arial', axisFontSize: 12, titleFontSize: 14, tickFontSize: 11, dpi: 300 },
  { name: 'IEEE', width: 600, height: 450, fontFamily: 'Times New Roman', axisFontSize: 12, titleFontSize: 14, tickFontSize: 11, dpi: 300 },
  { name: 'Presentation 16:9', width: 960, height: 540, fontFamily: 'Arial', axisFontSize: 16, titleFontSize: 20, tickFontSize: 14, dpi: 150 },
];

export function ExportPage() {
  const { graphStyle, setGraphStyle } = useAppStore();

  const applyPreset = (preset: typeof PRESETS[0]) => {
    setGraphStyle({
      width: preset.width,
      height: preset.height,
      fontFamily: preset.fontFamily,
      axisFontSize: preset.axisFontSize,
      titleFontSize: preset.titleFontSize,
      tickFontSize: preset.tickFontSize,
      dpi: preset.dpi,
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text mb-1">Export Settings</h2>
        <p className="text-sm text-text-muted">
          Configure chart export settings for publication-quality figures. Use the download buttons on each chart.
        </p>
      </div>

      {/* Presets */}
      <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
        <h3 className="text-sm font-medium text-text">Publication Presets</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => applyPreset(p)}
              className="px-3 py-2 bg-surface-alt border border-border rounded-lg text-xs text-text-muted hover:text-text hover:border-accent/30 transition-all text-left"
            >
              <p className="font-medium">{p.name}</p>
              <p className="text-text-dim">
                {p.width}×{p.height} · {p.dpi} DPI · {p.fontFamily}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Custom controls */}
      <div className="bg-surface rounded-xl border border-border p-4 space-y-4">
        <h3 className="text-sm font-medium text-text">Custom Settings</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs text-text-muted mb-1">Width (px)</label>
            <input
              type="number"
              value={graphStyle.width}
              onChange={(e) => setGraphStyle({ width: parseInt(e.target.value) || 600 })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Height (px)</label>
            <input
              type="number"
              value={graphStyle.height}
              onChange={(e) => setGraphStyle({ height: parseInt(e.target.value) || 450 })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">DPI</label>
            <select
              value={graphStyle.dpi}
              onChange={(e) => setGraphStyle({ dpi: parseInt(e.target.value) })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text"
            >
              <option value={150}>150</option>
              <option value={300}>300</option>
              <option value={600}>600</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Font Family</label>
            <select
              value={graphStyle.fontFamily}
              onChange={(e) => setGraphStyle({ fontFamily: e.target.value })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text"
            >
              <option value="Times New Roman">Times New Roman</option>
              <option value="Arial">Arial</option>
              <option value="Helvetica">Helvetica</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Axis Label Font Size</label>
            <input
              type="number"
              min={8}
              max={24}
              value={graphStyle.axisFontSize}
              onChange={(e) => setGraphStyle({ axisFontSize: parseInt(e.target.value) || 12 })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Tick Font Size</label>
            <input
              type="number"
              min={6}
              max={20}
              value={graphStyle.tickFontSize}
              onChange={(e) => setGraphStyle({ tickFontSize: parseInt(e.target.value) || 11 })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Title Font Size</label>
            <input
              type="number"
              min={10}
              max={28}
              value={graphStyle.titleFontSize}
              onChange={(e) => setGraphStyle({ titleFontSize: parseInt(e.target.value) || 14 })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Line Width</label>
            <input
              type="number"
              min={1}
              max={5}
              step={0.5}
              value={graphStyle.lineWidth}
              onChange={(e) => setGraphStyle({ lineWidth: parseFloat(e.target.value) || 2 })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Marker Size</label>
            <input
              type="number"
              min={1}
              max={10}
              value={graphStyle.markerSize}
              onChange={(e) => setGraphStyle({ markerSize: parseInt(e.target.value) || 4 })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Background</label>
            <input
              type="color"
              value={graphStyle.backgroundColor}
              onChange={(e) => setGraphStyle({ backgroundColor: e.target.value })}
              className="w-full h-[38px] bg-surface-alt border border-border rounded-lg cursor-pointer"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Axis Color</label>
            <input
              type="color"
              value={graphStyle.axisColor}
              onChange={(e) => setGraphStyle({ axisColor: e.target.value })}
              className="w-full h-[38px] bg-surface-alt border border-border rounded-lg cursor-pointer"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Grid Color</label>
            <input
              type="color"
              value={graphStyle.gridColor}
              onChange={(e) => setGraphStyle({ gridColor: e.target.value })}
              className="w-full h-[38px] bg-surface-alt border border-border rounded-lg cursor-pointer"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Border Color</label>
            <input
              type="color"
              value={graphStyle.borderColor}
              onChange={(e) => setGraphStyle({ borderColor: e.target.value })}
              className="w-full h-[38px] bg-surface-alt border border-border rounded-lg cursor-pointer"
            />
          </div>
          <div>
            <label className="block text-xs text-text-muted mb-1">Border Width</label>
            <input
              type="number"
              min={0}
              max={5}
              value={graphStyle.borderWidth}
              onChange={(e) => setGraphStyle({ borderWidth: parseInt(e.target.value) || 1 })}
              className="w-full bg-surface-alt border border-border rounded-lg px-3 py-2 text-sm text-text font-mono"
            />
          </div>
          <div className="flex items-center gap-2 pt-5">
            <input
              type="checkbox"
              checked={graphStyle.showGrid}
              onChange={(e) => setGraphStyle({ showGrid: e.target.checked })}
              className="w-4 h-4 accent-accent"
              id="grid-check"
            />
            <label htmlFor="grid-check" className="text-sm text-text-muted">Show Grid</label>
          </div>
          <div className="flex items-center gap-2 pt-5">
            <input
              type="checkbox"
              checked={graphStyle.showLegend}
              onChange={(e) => setGraphStyle({ showLegend: e.target.checked })}
              className="w-4 h-4 accent-accent"
              id="legend-check"
            />
            <label htmlFor="legend-check" className="text-sm text-text-muted">Show Legend</label>
          </div>
          <div className="flex items-center gap-2 pt-5">
            <input
              type="checkbox"
              checked={graphStyle.showBorder}
              onChange={(e) => setGraphStyle({ showBorder: e.target.checked })}
              className="w-4 h-4 accent-accent"
              id="border-check"
            />
            <label htmlFor="border-check" className="text-sm text-text-muted">Show Border</label>
          </div>
        </div>
      </div>

      {/* Grid opacity slider */}
      <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
        <h3 className="text-sm font-medium text-text">Grid Opacity</h3>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={graphStyle.gridOpacity}
            onChange={(e) => setGraphStyle({ gridOpacity: parseFloat(e.target.value) })}
            className="flex-1 accent-accent"
          />
          <span className="text-sm text-text-muted font-mono w-10">{graphStyle.gridOpacity.toFixed(2)}</span>
        </div>
      </div>

      <div className="bg-surface-alt rounded-xl border border-border p-4 text-sm text-text-muted">
        <p className="font-medium text-text mb-2">How to export charts</p>
        <p>
          Each chart throughout the app has SVG/PNG download buttons in the top-right corner.
          Configure your desired settings here, then use the download buttons on the Smoothing,
          Parameters, or ANN pages to export individual charts.
        </p>
      </div>
    </div>
  );
}
