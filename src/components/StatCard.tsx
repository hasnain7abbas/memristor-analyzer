interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
  quality?: 'good' | 'ok' | 'poor';
}

export function StatCard({ label, value, unit, color, quality }: StatCardProps) {
  const borderColor = quality === 'good'
    ? 'border-green/30'
    : quality === 'ok'
      ? 'border-amber/30'
      : quality === 'poor'
        ? 'border-red/30'
        : 'border-border';

  const valueColor = color || (quality === 'good'
    ? 'text-green'
    : quality === 'ok'
      ? 'text-amber'
      : quality === 'poor'
        ? 'text-red'
        : 'text-text');

  return (
    <div className={`bg-surface rounded-xl border ${borderColor} p-4`}>
      <p className="text-xs text-text-muted mb-1">{label}</p>
      <p className={`text-xl font-semibold font-mono ${valueColor}`}>
        {typeof value === 'number' ? value.toFixed(4) : value}
        {unit && <span className="text-sm text-text-dim ml-1">{unit}</span>}
      </p>
    </div>
  );
}
