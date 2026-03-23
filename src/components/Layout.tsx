import { useState } from 'react';
import { Upload, SlidersHorizontal, Calculator, Brain, Download, HelpCircle } from 'lucide-react';
import { useAppStore } from '../stores/useAppStore';
import { HelpDialog } from './HelpDialog';
import type { TabId } from '../types';

const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'upload', label: 'Upload', icon: <Upload size={16} /> },
  { id: 'smoothing', label: 'Smoothing', icon: <SlidersHorizontal size={16} /> },
  { id: 'parameters', label: 'Parameters', icon: <Calculator size={16} /> },
  { id: 'ann', label: 'ANN / MNIST', icon: <Brain size={16} /> },
  { id: 'export', label: 'Export', icon: <Download size={16} /> },
];

export function Layout({ children }: { children: React.ReactNode }) {
  const { activeTab, setActiveTab, uploadedTests } = useAppStore();
  const hasData = Object.keys(uploadedTests).length > 0;
  const [showHelp, setShowHelp] = useState(false);

  return (
    <div className="min-h-screen bg-bg">
      {/* Top Nav */}
      <nav className="sticky top-0 z-50 bg-surface border-b border-border px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
            <Brain size={18} className="text-accent" />
          </div>
          <h1 className="text-lg font-semibold text-text">Memristor Neural Analyzer</h1>
        </div>

        <div className="flex gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-accent/15 text-accent'
                  : 'text-text-muted hover:text-text hover:bg-surface-alt'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowHelp(true)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-cyan hover:bg-cyan/10 transition-colors"
            title="Formula Reference & Help"
          >
            <HelpCircle size={16} />
            Help
          </button>
          {hasData && (
            <span className="text-xs px-2 py-1 bg-green/10 text-green rounded-full">
              {Object.keys(uploadedTests).length} dataset(s)
            </span>
          )}
        </div>
      </nav>

      {/* Content — wider container for larger graphs */}
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        {children}
      </main>

      {/* Help Dialog */}
      {showHelp && <HelpDialog onClose={() => setShowHelp(false)} />}
    </div>
  );
}
