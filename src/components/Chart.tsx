import { useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts';
import { Download } from 'lucide-react';
import { useAppStore } from '../stores/useAppStore';
import type { PlotType } from '../types';

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
  caption?: string;
  plotType?: PlotType;
  heightOverride?: number;
  id?: string;
}

export function AppLineChart({
  data, lines, xKey, xLabel, yLabel, title, caption, plotType = 'line', heightOverride, id,
}: LineChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const gs = useAppStore((s) => s.graphStyle);

  const chartHeight = heightOverride ?? gs.height;

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
    const scale = gs.dpi / 96;
    const w = gs.width * scale;
    const h = chartHeight * scale;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      ctx.fillStyle = gs.backgroundColor;
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
  }, [id, gs, chartHeight]);

  const getLineType = (pt: PlotType): 'monotone' | 'stepAfter' => {
    if (pt === 'step') return 'stepAfter';
    return 'monotone';
  };

  const showDots = (line: ChartLine): boolean | object => {
    if (plotType === 'scatter') return { r: gs.markerSize };
    if (plotType === 'line_scatter') return { r: gs.markerSize };
    if (line.dot === true) return { r: gs.markerSize };
    if (line.dot === false) return false;
    return { r: gs.markerSize };
  };

  const getStrokeWidth = (): number => {
    if (plotType === 'scatter') return 0;
    return gs.lineWidth;
  };

  const borderStyle = gs.showBorder
    ? `${gs.borderWidth}px solid ${gs.borderColor}`
    : 'none';

  return (
    <div className="space-y-2" style={{ fontFamily: gs.fontFamily }}>
      {title && (
        <div className="flex items-center justify-between">
          <h3 style={{ fontSize: gs.titleFontSize, color: gs.axisColor, fontFamily: gs.fontFamily }} className="font-semibold">
            {title}
          </h3>
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
      <div
        ref={chartRef}
        style={{
          backgroundColor: gs.backgroundColor,
          border: borderStyle,
          borderRadius: '8px',
          padding: '16px',
          fontFamily: gs.fontFamily,
        }}
      >
        <ResponsiveContainer width="100%" height={chartHeight}>
          <LineChart data={data}>
            {gs.showGrid && (
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={gs.gridColor}
                opacity={gs.gridOpacity}
              />
            )}
            <XAxis
              dataKey={xKey}
              stroke={gs.axisColor}
              tick={{ fill: gs.axisColor, fontSize: gs.tickFontSize, fontFamily: gs.fontFamily }}
              label={xLabel ? { value: xLabel, fill: gs.axisColor, dy: 15, fontSize: gs.axisFontSize, fontFamily: gs.fontFamily } : undefined}
            />
            <YAxis
              stroke={gs.axisColor}
              tick={{ fill: gs.axisColor, fontSize: gs.tickFontSize, fontFamily: gs.fontFamily }}
              label={yLabel ? { value: yLabel, fill: gs.axisColor, angle: -90, dx: -20, fontSize: gs.axisFontSize, fontFamily: gs.fontFamily } : undefined}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: gs.backgroundColor,
                border: `1px solid ${gs.gridColor}`,
                borderRadius: '8px',
                fontSize: gs.tickFontSize,
                fontFamily: gs.fontFamily,
              }}
            />
            {gs.showLegend && (
              <Legend wrapperStyle={{ fontSize: gs.axisFontSize, fontFamily: gs.fontFamily }} />
            )}
            {lines.map((line) => (
              <Line
                key={line.dataKey}
                type={getLineType(plotType)}
                dataKey={line.dataKey}
                stroke={line.color}
                name={line.name}
                dot={showDots(line)}
                strokeWidth={getStrokeWidth()}
                legendType={plotType === 'scatter' ? 'circle' : undefined}
                strokeDasharray={
                  line.type === 'dashed' ? '6 3' : line.type === 'dotted' ? '2 3' : undefined
                }
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {caption && (
        <p style={{ fontSize: gs.axisFontSize, color: gs.axisColor, fontFamily: gs.fontFamily }} className="text-center italic">
          {caption}
        </p>
      )}
    </div>
  );
}
