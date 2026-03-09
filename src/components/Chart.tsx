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

  const getSvgString = useCallback(() => {
    if (!chartRef.current) return null;
    const svg = chartRef.current.querySelector('svg');
    if (!svg) return null;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    clone.setAttribute('width', String(gs.width));
    clone.setAttribute('height', String(chartHeight));
    const serializer = new XMLSerializer();
    return serializer.serializeToString(clone);
  }, [gs.width, chartHeight]);

  const handleExportSVG = useCallback(() => {
    const svgStr = getSvgString();
    if (!svgStr) return;
    const blob = new Blob([svgStr], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${id || 'chart'}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  }, [id, getSvgString]);

  const handleExportPNG = useCallback(() => {
    const svgStr = getSvgString();
    if (!svgStr) return;
    const scale = gs.dpi / 96;
    const w = gs.width * scale;
    const h = chartHeight * scale;

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    // Use data URI to avoid canvas tainting from blob URLs
    const b64 = btoa(unescape(encodeURIComponent(svgStr)));
    const dataUri = `data:image/svg+xml;base64,${b64}`;
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
    };
    img.src = dataUri;
  }, [id, gs, chartHeight, getSvgString]);

  const getLineType = (pt: PlotType): 'monotone' | 'stepAfter' => {
    if (pt === 'step') return 'stepAfter';
    return 'monotone';
  };

  const showDots = (line: ChartLine): boolean | object => {
    if (plotType === 'scatter') return { r: gs.markerSize };
    if (plotType === 'line_scatter') return { r: gs.markerSize };
    if (line.dot === true) return { r: Math.max(1, gs.markerSize - 1) };
    if (line.dot === false) return false;
    // Default: no dots for line/step plots (avoids clutter with many data points)
    if (plotType === 'line' || plotType === 'step') return false;
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
      <div className="flex items-center justify-between">
        {title ? (
          <h3 style={{ fontSize: gs.titleFontSize, color: gs.axisColor, fontFamily: gs.fontFamily }} className="font-semibold">
            {title}
          </h3>
        ) : <div />}
        <div className="flex gap-1">
          <button
            onClick={handleExportSVG}
            className="flex items-center gap-1 px-2 py-1 rounded text-text-dim hover:text-text hover:bg-surface-alt text-xs"
            title="Export SVG"
          >
            <Download size={12} />
            <span>SVG</span>
          </button>
          <button
            onClick={handleExportPNG}
            className="flex items-center gap-1 px-2 py-1 rounded text-text-dim hover:text-text hover:bg-surface-alt text-xs"
            title="Export PNG"
          >
            <Download size={12} />
            <span>PNG</span>
          </button>
        </div>
      </div>
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
                connectNulls={false}
                isAnimationActive={false}
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
