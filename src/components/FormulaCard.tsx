import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { Tex } from './Tex';
import type { FormulaSection } from '../types';

interface FormulaCardProps {
  title: string;
  formula: string;
  latex?: string;
  explanation: string;
  example?: string;
  highlight?: boolean;
  sections?: FormulaSection[];
  physicalMeaning?: string;
  reference?: string;
}

export function FormulaCard({ title, formula, latex, explanation, example, highlight, sections, physicalMeaning, reference }: FormulaCardProps) {
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
          {/* Main formula: render with KaTeX if latex is provided */}
          {latex ? (
            <div className="bg-bg/60 p-4 rounded-lg overflow-x-auto text-center">
              <Tex math={latex} display />
            </div>
          ) : (
            <pre className="font-mono text-sm text-cyan bg-bg/60 p-3 rounded-lg overflow-x-auto whitespace-pre-wrap">
              {formula}
            </pre>
          )}
          <p className="text-sm text-text-muted leading-relaxed">{explanation}</p>

          {sections && sections.length > 0 && (
            <div className="space-y-2 mt-2">
              {sections.map((sec, i) => (
                <div key={i} className="border-l-2 border-accent/30 pl-3">
                  <p className="text-xs font-medium text-accent mb-1">{sec.subtitle}</p>
                  {sec.latex ? (
                    <div className="bg-bg/40 p-2 rounded overflow-x-auto text-center">
                      <Tex math={sec.latex} display />
                    </div>
                  ) : (
                    <pre className="font-mono text-xs text-text-muted whitespace-pre-wrap bg-bg/40 p-2 rounded">
                      {sec.content}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          )}

          {physicalMeaning && (
            <div className="bg-purple/5 border border-purple/20 rounded-lg p-3">
              <p className="text-xs font-medium text-purple mb-1">Physical Meaning</p>
              <p className="text-xs text-text-muted leading-relaxed">{physicalMeaning}</p>
            </div>
          )}

          {example && (
            <div className="bg-green/5 border border-green/20 rounded-lg p-3">
              <p className="text-xs font-medium text-green mb-1">Worked Example</p>
              <pre className="font-mono text-xs text-text-muted whitespace-pre-wrap">
                {example}
              </pre>
            </div>
          )}

          {reference && (
            <p className="text-[11px] text-text-dim italic">
              Ref: {reference}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
