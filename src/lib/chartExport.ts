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
