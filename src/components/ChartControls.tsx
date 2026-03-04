import type { PlotType, ChartLocalSettings } from '../types';

interface ChartControlsProps {
  settings: ChartLocalSettings;
  onChange: (s: Partial<ChartLocalSettings>) => void;
}

const PLOT_OPTIONS: { value: PlotType; label: string }[] = [
  { value: 'line', label: 'Line' },
  { value: 'scatter', label: 'Scatter' },
  { value: 'line_scatter', label: 'Line + Scatter' },
  { value: 'step', label: 'Step' },
];

export function ChartControls({ settings, onChange }: ChartControlsProps) {
  return (
    <div className="flex flex-wrap items-end gap-3 text-xs">
      <div>
        <label className="block text-text-muted mb-0.5">X Label</label>
        <input
          type="text"
          value={settings.xLabel}
          onChange={(e) => onChange({ xLabel: e.target.value })}
          className="bg-surface-alt border border-border rounded px-2 py-1 text-text w-36"
        />
      </div>
      <div>
        <label className="block text-text-muted mb-0.5">Y Label</label>
        <input
          type="text"
          value={settings.yLabel}
          onChange={(e) => onChange({ yLabel: e.target.value })}
          className="bg-surface-alt border border-border rounded px-2 py-1 text-text w-36"
        />
      </div>
      <div>
        <label className="block text-text-muted mb-0.5">Plot Type</label>
        <select
          value={settings.plotType}
          onChange={(e) => onChange({ plotType: e.target.value as PlotType })}
          className="bg-surface-alt border border-border rounded px-2 py-1 text-text"
        >
          {PLOT_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>
      <div>
        <label className="block text-text-muted mb-0.5">Caption</label>
        <input
          type="text"
          value={settings.caption}
          onChange={(e) => onChange({ caption: e.target.value })}
          placeholder="Figure caption..."
          className="bg-surface-alt border border-border rounded px-2 py-1 text-text w-48"
        />
      </div>
    </div>
  );
}
