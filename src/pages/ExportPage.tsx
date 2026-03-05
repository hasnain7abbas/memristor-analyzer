import { useAppStore } from '../stores/useAppStore';
import { AppLineChart } from '../components/Chart';

const PRESETS = [
  { name: 'Frontiers/Elsevier (single col)', width: 600, height: 450, fontFamily: 'Times New Roman', axisFontSize: 12, titleFontSize: 14, tickFontSize: 11, dpi: 300 },
  { name: 'Frontiers/Elsevier (double col)', width: 1200, height: 450, fontFamily: 'Times New Roman', axisFontSize: 14, titleFontSize: 16, tickFontSize: 12, dpi: 300 },
  { name: 'Nature/Science', width: 600, height: 500, fontFamily: 'Helvetica', axisFontSize: 11, titleFontSize: 13, tickFontSize: 10, dpi: 600 },
  { name: 'ACS Journals', width: 600, height: 450, fontFamily: 'Arial', axisFontSize: 12, titleFontSize: 14, tickFontSize: 11, dpi: 300 },
  { name: 'IEEE', width: 600, height: 450, fontFamily: 'Times New Roman', axisFontSize: 12, titleFontSize: 14, tickFontSize: 11, dpi: 300 },
  { name: 'Presentation 16:9', width: 960, height: 540, fontFamily: 'Arial', axisFontSize: 16, titleFontSize: 20, tickFontSize: 14, dpi: 150 },
];

export function ExportPage() {
  const { graphStyle, setGraphStyle, extractedParams, annResults, uploadedTests, smoothedData } = useAppStore();

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

  // Build chart data for gallery
  const pdChartData = extractedParams
    ? [
        ...extractedParams.potentiationSmoothed.map((v: number, i: number) => ({
          index: i + 1,
          potSmoothed: v as number | undefined,
          potFitted: extractedParams.potentiationFitted[i] as number | undefined,
          depSmoothed: undefined as number | undefined,
          depFitted: undefined as number | undefined,
        })),
        ...extractedParams.depressionSmoothed.map((v: number, i: number) => ({
          index: extractedParams.potentiationSmoothed.length + i + 1,
          potSmoothed: undefined as number | undefined,
          potFitted: undefined as number | undefined,
          depSmoothed: v as number | undefined,
          depFitted: extractedParams.depressionFitted[i] as number | undefined,
        })),
      ]
    : [];

  const deltaGData = extractedParams
    ? extractedParams.deltaG.map((d: { G: number; dG: number }) => ({ G: d.G, dG: d.dG }))
    : [];

  const accChartData = annResults.map((r) => ({
    epoch: r.epoch,
    ideal: r.idealAccuracy,
    memristor: r.memristorAccuracy,
  }));

  // Build smoothing chart from first available
  const testIds = Object.keys(uploadedTests);
  let smoothingChartData: Record<string, unknown>[] = [];
  let smoothingColName = '';
  if (testIds.length > 0) {
    const firstTest = uploadedTests[testIds[0]];
    const numCols = firstTest.dataset.columns.filter((c) => c.values.length > 0);
    const condCol = numCols.find((c) => /conductance|cond|g_?us|^g$/i.test(c.name)) || numCols[0];
    if (condCol) {
      smoothingColName = condCol.name;
      const key = `${testIds[0]}_${condCol.name}`;
      const smooth = smoothedData[key] || [];
      smoothingChartData = condCol.values.map((val, i) => ({
        index: i + 1,
        raw: val,
        smoothed: smooth[i] ?? val,
      }));
    }
  }

  const hasAnyChart = pdChartData.length > 0 || deltaGData.length > 0 || accChartData.length > 0 || smoothingChartData.length > 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text mb-1">Export Settings</h2>
        <p className="text-sm text-text-muted">
          Configure chart export settings for publication-quality figures. Each chart below has SVG/PNG download buttons.
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

      {/* Chart Gallery */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-text">Chart Gallery — Preview & Export</h3>
        <p className="text-xs text-text-muted">
          All available charts are rendered below with current export settings. Use the SVG/PNG buttons on each chart to download.
        </p>

        {!hasAnyChart && (
          <div className="bg-surface-alt rounded-xl border border-border p-8 text-center text-text-dim text-sm">
            No charts available yet. Upload data and extract parameters to see charts here.
          </div>
        )}

        {smoothingChartData.length > 0 && (
          <AppLineChart
            data={smoothingChartData}
            lines={[
              { dataKey: 'raw', color: '#8494b2', name: 'Raw', dot: true, type: 'dotted' },
              { dataKey: 'smoothed', color: '#4f8ff7', name: 'Smoothed' },
            ]}
            xKey="index"
            xLabel="Pulse Number"
            yLabel={smoothingColName}
            title="Raw vs Smoothed Data"
            id="export-smoothing"
          />
        )}

        {pdChartData.length > 0 && (
          <AppLineChart
            data={pdChartData}
            lines={[
              { dataKey: 'potSmoothed', color: '#34d399', name: 'P Smoothed' },
              { dataKey: 'potFitted', color: '#34d399', name: 'P Fitted', type: 'dashed' },
              { dataKey: 'depSmoothed', color: '#f87171', name: 'D Smoothed' },
              { dataKey: 'depFitted', color: '#f87171', name: 'D Fitted', type: 'dashed' },
            ]}
            xKey="index"
            xLabel="Pulse Number"
            yLabel="Conductance (µS)"
            title="Potentiation / Depression Curves"
            id="export-pd-curves"
          />
        )}

        {deltaGData.length > 0 && (
          <AppLineChart
            data={deltaGData}
            lines={[{ dataKey: 'dG', color: '#a78bfa', name: 'ΔG' }]}
            xKey="G"
            xLabel="G (µS)"
            yLabel="ΔG (µS)"
            title="ΔG vs G — Switching Statistics"
            plotType="scatter"
            id="export-delta-g"
          />
        )}

        {accChartData.length > 0 && (
          <AppLineChart
            data={accChartData}
            lines={[
              { dataKey: 'ideal', color: '#4f8ff7', name: 'Ideal' },
              { dataKey: 'memristor', color: '#f87171', name: 'Memristor' },
            ]}
            xKey="epoch"
            xLabel="Epoch"
            yLabel="Accuracy (%)"
            title="ANN Accuracy vs Epoch"
            id="export-ann-accuracy"
          />
        )}
      </div>
    </div>
  );
}
