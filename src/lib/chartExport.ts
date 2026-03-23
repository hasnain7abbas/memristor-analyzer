import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';

// Chart export utilities
export function exportSVGFromContainer(container: HTMLElement, filename: string) {
  const svg = container.querySelector('svg');
  if (!svg) return;
  const serializer = new XMLSerializer();
  const svgStr = serializer.serializeToString(svg);
  const blob = new Blob([svgStr], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.svg`;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportPNGFromContainer(
  container: HTMLElement,
  filename: string,
  dpi = 300,
  bgColor = '#ffffff',
) {
  const svg = container.querySelector('svg');
  if (!svg) return;
  const serializer = new XMLSerializer();
  const svgStr = serializer.serializeToString(svg);
  const scale = dpi / 96;
  const rect = svg.getBoundingClientRect();
  const w = rect.width * scale;
  const h = rect.height * scale;

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const img = new Image();
  const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  img.onload = () => {
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);
    canvas.toBlob((b) => {
      if (!b) return;
      const u = URL.createObjectURL(b);
      const a = document.createElement('a');
      a.href = u;
      a.download = `${filename}.png`;
      a.click();
      URL.revokeObjectURL(u);
    }, 'image/png');
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

/**
 * Convert an array of objects to delimited text (CSV or tab-delimited).
 * Columns are derived from the keys of the first object.
 * Only includes columns that have at least one numeric or string value.
 */
function toDelimited(
  data: Record<string, any>[],
  columns: string[],
  delimiter: string,
): string {
  const header = columns.join(delimiter);
  const rows = data.map((row) =>
    columns.map((col) => {
      const v = row[col];
      if (v === undefined || v === null) return '';
      return typeof v === 'number' ? v.toString() : String(v);
    }).join(delimiter)
  );
  return [header, ...rows].join('\n');
}

/**
 * Export chart data via Tauri save dialog.
 * @param data - Array of row objects (same structure as chart data)
 * @param columns - Column names to include (in order)
 * @param defaultName - Default filename (without extension)
 * @param format - 'csv' or 'txt' (tab-delimited)
 */
export async function exportChartData(
  data: Record<string, any>[],
  columns: string[],
  defaultName: string,
  format: 'csv' | 'txt' = 'txt',
): Promise<void> {
  const delimiter = format === 'csv' ? ',' : '\t';
  const ext = format === 'csv' ? 'csv' : 'txt';
  const filterName = format === 'csv' ? 'CSV File' : 'Tab-Delimited Text';

  const path = await save({
    defaultPath: `${defaultName}.${ext}`,
    filters: [
      { name: filterName, extensions: [ext] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });

  if (!path) return;

  const content = toDelimited(data, columns, delimiter);
  await writeTextFile(path, content);
}

/**
 * Export multiple datasets into one file with section headers.
 * Useful for P/D curves where potentiation and depression are separate.
 */
export async function exportMultiSection(
  sections: { label: string; columns: string[]; data: Record<string, any>[] }[],
  defaultName: string,
  format: 'csv' | 'txt' = 'txt',
): Promise<void> {
  const delimiter = format === 'csv' ? ',' : '\t';
  const ext = format === 'csv' ? 'csv' : 'txt';
  const filterName = format === 'csv' ? 'CSV File' : 'Tab-Delimited Text';

  const path = await save({
    defaultPath: `${defaultName}.${ext}`,
    filters: [
      { name: filterName, extensions: [ext] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });

  if (!path) return;

  const parts: string[] = [];
  for (const sec of sections) {
    if (format === 'txt') {
      parts.push(`# ${sec.label}`);
    }
    parts.push(toDelimited(sec.data, sec.columns, delimiter));
    parts.push(''); // blank line between sections
  }

  await writeTextFile(path, parts.join('\n'));
}
