import { useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter,
} from 'recharts';
import { Download } from 'lucide-react';
import type { GraphStyle } from '../types';

interface ChartLine {
  dataKey: string;
  color: string;
  name: string;
  type?: 'solid' | 'dashed' | 'dotted';
  dot?: boolean;
}

interface LineChartProps {
  data: Record<string, unknown>[];
  lines: ChartLine[];
  xKey: string;
  xLabel?: string;
  yLabel?: string;
  title?: string;
  style?: Partial<GraphStyle>;
  id?: string;
}

export function AppLineChart({
  data, lines, xKey, xLabel, yLabel, title, style, id,
}: LineChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  const handleExportSVG = useCallback(() => {
    if (!chartRef.current) return;
    const svg = chartRef.current.querySelector('svg');
    if (!svg) return;
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svg);
    const blob = new Blob([svgStr], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${id || 'chart'}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  }, [id]);

  const handleExportPNG = useCallback(() => {
    if (!chartRef.current) return;
    const svg = chartRef.current.querySelector('svg');
    if (!svg) return;
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svg);
    const dpi = style?.dpi || 300;
    const scale = dpi / 96;
    const w = (style?.width || 600) * scale;
    const h = (style?.height || 400) * scale;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      ctx.fillStyle = style?.backgroundColor || '#ffffff';
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob((b) => {
        if (!b) return;
        const u = URL.createObjectURL(b);
        const a = document.createElement('a');
        a.href = u;
        a.download = `${id || 'chart'}.png`;
        a.click();
        URL.revokeObjectURL(u);
      }, 'image/png');
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }, [id, style]);

  return (
    <div className="space-y-2">
      {title && (
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-text">{title}</h3>
          <div className="flex gap-1">
            <button
              onClick={handleExportSVG}
              className="p-1.5 rounded text-text-dim hover:text-text hover:bg-surface-alt"
              title="Export SVG"
            >
              <Download size={14} />
            </button>
            <button
              onClick={handleExportPNG}
              className="p-1.5 rounded text-text-dim hover:text-text hover:bg-surface-alt"
              title="Export PNG"
            >
              <Download size={14} />
            </button>
          </div>
        </div>
      )}
      <div ref={chartRef} className="bg-surface rounded-lg p-4 border border-border">
        <ResponsiveContainer width="100%" height={style?.height || 350}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1c2840" opacity={0.5} />
            <XAxis
              dataKey={xKey}
              stroke="#4a5c7a"
              tick={{ fill: '#8494b2', fontSize: 11 }}
              label={xLabel ? { value: xLabel, fill: '#8494b2', dy: 15, fontSize: 12 } : undefined}
            />
            <YAxis
              stroke="#4a5c7a"
              tick={{ fill: '#8494b2', fontSize: 11 }}
              label={yLabel ? { value: yLabel, fill: '#8494b2', angle: -90, dx: -20, fontSize: 12 } : undefined}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d1219',
                border: '1px solid #1c2840',
                borderRadius: '8px',
                fontSize: 12,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {lines.map((line) => (
              <Line
                key={line.dataKey}
                type="monotone"
                dataKey={line.dataKey}
                stroke={line.color}
                name={line.name}
                dot={line.dot !== false ? { r: 2 } : false}
                strokeWidth={2}
                strokeDasharray={
                  line.type === 'dashed' ? '6 3' : line.type === 'dotted' ? '2 3' : undefined
                }
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

interface ScatterChartProps {
  data: { x: number; y: number }[];
  xLabel?: string;
  yLabel?: string;
  title?: string;
  color?: string;
  id?: string;
}

export function AppScatterChart({
  data, xLabel, yLabel, title, color = '#4f8ff7', id: _id,
}: ScatterChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  return (
    <div className="space-y-2">
      {title && <h3 className="text-sm font-semibold text-text">{title}</h3>}
      <div ref={chartRef} className="bg-surface rounded-lg p-4 border border-border">
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#1c2840" opacity={0.5} />
            <XAxis
              dataKey="x"
              type="number"
              stroke="#4a5c7a"
              tick={{ fill: '#8494b2', fontSize: 11 }}
              label={xLabel ? { value: xLabel, fill: '#8494b2', dy: 15, fontSize: 12 } : undefined}
              name={xLabel}
            />
            <YAxis
              dataKey="y"
              type="number"
              stroke="#4a5c7a"
              tick={{ fill: '#8494b2', fontSize: 11 }}
              label={yLabel ? { value: yLabel, fill: '#8494b2', angle: -90, dx: -20, fontSize: 12 } : undefined}
              name={yLabel}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d1219',
                border: '1px solid #1c2840',
                borderRadius: '8px',
                fontSize: 12,
              }}
            />
            <Scatter data={data} fill={color} r={3} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
