import { X, ExternalLink } from 'lucide-react';

interface AboutDialogProps {
  onClose: () => void;
}

export function AboutDialog({ onClose }: AboutDialogProps) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm overflow-y-auto py-8">
      <div className="w-full max-w-2xl bg-surface border border-border rounded-2xl shadow-2xl mx-4">
        {/* Header */}
        <div className="bg-surface border-b border-border rounded-t-2xl px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-text">About Memristor Neural Analyzer</h1>
          <button onClick={onClose} className="p-2 hover:bg-surface-alt rounded-lg transition-colors">
            <X size={20} className="text-text-muted" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-6 space-y-6">
          {/* Project Info */}
          <div className="space-y-2">
            <p className="text-sm text-text-muted leading-relaxed">
              Desktop application for memristor device characterization and neuromorphic ANN simulation.
              Built for researchers working on resistive switching devices for neuromorphic computing applications.
              Upload experimental potentiation/depression data, extract device parameters, simulate neural network
              performance with realistic device non-idealities, and export publication-quality figures.
            </p>
          </div>

          {/* Developer */}
          <div className="bg-bg/60 rounded-xl border border-border/50 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-text">Developer</p>
                <p className="text-lg font-semibold text-accent">Hasnain Abbas</p>
              </div>
              <a
                href="https://github.com/hasnain7abbas"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-cyan hover:bg-cyan/10 transition-colors"
              >
                <ExternalLink size={14} />
                github.com/hasnain7abbas
              </a>
            </div>
            <div className="flex items-center gap-4 text-xs text-text-muted">
              <span>License: MIT</span>
            </div>
          </div>

          {/* References */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-purple">References &amp; Acknowledgments</h3>
            <div className="space-y-2 text-xs text-text-muted leading-relaxed">
              <div className="bg-bg/40 rounded-lg p-3 space-y-2">
                <p>
                  <strong className="text-text">Non-linearity model:</strong> Based on the exponential kinetic model
                  widely used in memristive device characterization.
                  <br />
                  <span className="italic text-text-dim">G. W. Burr et al., "Neuromorphic computing using non-volatile memory,"
                  Advances in Physics: X, vol. 2, no. 1, 2017.</span>
                </p>
                <p>
                  <strong className="text-text">ANN simulation methodology:</strong> Copy-and-degrade approach with
                  snap-to-nearest-level quantization and Gaussian noise injection.
                  <br />
                  <span className="italic text-text-dim">G. W. Burr et al., "Experimental demonstration and tolerancing of a
                  large-scale neural network (165,000 synapses) using phase-change memory as the synaptic weight element,"
                  IEDM, 2015.</span>
                </p>
                <p>
                  <strong className="text-text">Alpha-accuracy mapping:</strong> Calibrated against systematic study of
                  121 combinations of non-linearity parameters.
                  <br />
                  <span className="italic text-text-dim">S. Kim et al., "Spiking neural network with memristor synapses having
                  non-linear weight update," Front. Comput. Neurosci. 15:646125, 2021.</span>
                </p>
                <p>
                  <strong className="text-text">PPF model:</strong> Paired-pulse facilitation with double exponential
                  decay for short-term synaptic plasticity characterization.
                  <br />
                  <span className="italic text-text-dim">R. Zucker &amp; W. Regehr, "Short-term synaptic plasticity,"
                  Annu. Rev. Physiol., vol. 64, pp. 355-405, 2002.</span>
                </p>
                <p>
                  <strong className="text-text">Memristor fundamentals:</strong>
                  <br />
                  <span className="italic text-text-dim">L. Chua, "Memristor -- The Missing Circuit Element,"
                  IEEE Trans. Circuit Theory, vol. 18, no. 5, pp. 507-519, 1971.</span>
                  <br />
                  <span className="italic text-text-dim">D. B. Strukov et al., "The missing memristor found,"
                  Nature, vol. 453, pp. 80-83, 2008.</span>
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-border px-6 py-3 text-xs text-text-dim text-center">
          Memristor Neural Analyzer v1.0.6
        </div>
      </div>
    </div>
  );
}
