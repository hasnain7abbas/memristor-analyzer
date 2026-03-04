import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface FormulaCardProps {
  title: string;
  formula: string;
  explanation: string;
  example?: string;
  highlight?: boolean;
}

export function FormulaCard({ title, formula, explanation, example, highlight }: FormulaCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`rounded-xl border transition-all ${
        highlight ? 'border-cyan/30 bg-cyan/5' : 'border-border bg-surface'
      }`}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left"
      >
        {expanded ? (
          <ChevronDown size={16} className="text-text-dim shrink-0" />
        ) : (
          <ChevronRight size={16} className="text-text-dim shrink-0" />
        )}
        <span className="text-sm font-medium text-text">{title}</span>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          <pre className="font-mono text-sm text-cyan bg-bg/60 p-3 rounded-lg overflow-x-auto whitespace-pre-wrap">
            {formula}
          </pre>
          <p className="text-sm text-text-muted leading-relaxed">{explanation}</p>
          {example && (
            <div className="bg-green/5 border border-green/20 rounded-lg p-3">
              <p className="text-xs font-medium text-green mb-1">Worked Example</p>
              <pre className="font-mono text-xs text-text-muted whitespace-pre-wrap">
                {example}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
