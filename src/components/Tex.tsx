import katex from 'katex';

interface TexProps {
  math: string;
  display?: boolean;
  className?: string;
}

/** Renders a LaTeX math expression using KaTeX. */
export function Tex({ math, display = false, className }: TexProps) {
  const html = katex.renderToString(math, {
    throwOnError: false,
    displayMode: display,
  });
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
