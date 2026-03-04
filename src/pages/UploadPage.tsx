import { useState } from 'react';
import {
  Zap, TrendingUp, RotateCcw, Layers, Activity, GitBranch, Radio, Repeat, ChevronDown, ChevronRight,
} from 'lucide-react';
import { useAppStore } from '../stores/useAppStore';
import { FileUploader } from '../components/FileUploader';
import { DataTable } from '../components/DataTable';
import { AppLineChart } from '../components/Chart';
import type { Dataset, TestType, ColumnMapping } from '../types';

const TEST_TYPES: TestType[] = [
  {
    id: 'pd_training',
    name: 'Potentiation / Depression',
    icon: 'zap',
    critical: true,
    description: 'Conductance response to sequential write pulses. Required for α extraction and ANN simulation.',
    requiredColumns: ['pulse_number', 'conductance_uS'],
    optionalColumns: ['cycle', 'type'],
    recordingGuide: 'Apply sequential potentiation pulses (e.g., +2V, 100µs) recording read current at V_read after each. Then apply depression pulses (e.g., -2V, 100µs).',
  },
  {
    id: 'ltp_ltd',
    name: 'LTP / LTD Linearity',
    icon: 'trending',
    critical: true,
    description: 'Long-term potentiation/depression for weight update linearity assessment.',
    requiredColumns: ['pulse_number', 'conductance_uS'],
    optionalColumns: ['type'],
    recordingGuide: 'Same as P/D but with optimized pulse parameters for linearity.',
  },
  {
    id: 'endurance',
    name: 'Endurance',
    icon: 'rotate',
    critical: false,
    description: 'HRS/LRS conductance over many switching cycles.',
    requiredColumns: ['cycle', 'g_hrs_uS', 'g_lrs_uS'],
    optionalColumns: [],
    recordingGuide: 'Record G_HRS and G_LRS at V_read after each set/reset cycle.',
  },
  {
    id: 'multilevel',
    name: 'Multi-Level States',
    icon: 'layers',
    critical: false,
    description: 'Distinct conductance levels for multi-bit storage.',
    requiredColumns: ['time_s', 'conductance_uS'],
    optionalColumns: ['level'],
    recordingGuide: 'Apply increasing pulse amplitudes and read conductance at each level.',
  },
  {
    id: 'epsc',
    name: 'EPSC / IPSC',
    icon: 'activity',
    critical: false,
    description: 'Excitatory/inhibitory post-synaptic current response.',
    requiredColumns: ['time_s', 'current_uA'],
    optionalColumns: [],
    recordingGuide: 'Apply single pulse and record current decay over time at V_read.',
  },
  {
    id: 'ppf',
    name: 'PPF / PPD',
    icon: 'gitbranch',
    critical: false,
    description: 'Paired-pulse facilitation: current ratio vs inter-pulse interval.',
    requiredColumns: ['interval_ms', 'A1_uA', 'A2_uA'],
    optionalColumns: [],
    recordingGuide: 'Apply two identical pulses with varying interval. Record peak current of each.',
  },
  {
    id: 'sddp',
    name: 'SDDP',
    icon: 'radio',
    critical: false,
    description: 'Spike duration-dependent plasticity.',
    requiredColumns: ['pulse_number', 'conductance_uS', 'duration_ms'],
    optionalColumns: [],
    recordingGuide: 'Vary pulse width while keeping amplitude constant. Record conductance change.',
  },
  {
    id: 'srdp',
    name: 'SRDP',
    icon: 'repeat',
    critical: false,
    description: 'Spike rate-dependent plasticity.',
    requiredColumns: ['pulse_number', 'conductance_uS', 'frequency_Hz'],
    optionalColumns: [],
    recordingGuide: 'Vary pulse frequency while keeping amplitude/width constant.',
  },
  {
    id: 'bipolar',
    name: 'Bipolar I-V Sweeps',
    icon: 'repeat',
    critical: false,
    description: 'Current-voltage sweep characteristics for resistive switching.',
    requiredColumns: ['voltage_V', 'current_uA'],
    optionalColumns: ['sweep'],
    recordingGuide: 'Sweep voltage: 0 → +V_max → 0 → -V_max → 0. Record current.',
  },
];

const iconMap: Record<string, React.ReactNode> = {
  zap: <Zap size={18} />,
  trending: <TrendingUp size={18} />,
  rotate: <RotateCcw size={18} />,
  layers: <Layers size={18} />,
  activity: <Activity size={18} />,
  gitbranch: <GitBranch size={18} />,
  radio: <Radio size={18} />,
  repeat: <Repeat size={18} />,
};

function autoMapColumns(headers: string[], required: string[]): ColumnMapping {
  const mapping: ColumnMapping = {};
  const patterns: Record<string, RegExp> = {
    conductance_uS: /conductance|cond|g_?us|^g$/i,
    current_uA: /current|i_?ua/i,
    pulse_number: /pulse|num|#|^n$/i,
    type: /type|phase/i,
    cycle: /cycle/i,
    time_s: /time|^t$/i,
    voltage_V: /voltage|^v$/i,
    interval_ms: /interval|delay/i,
    A1_uA: /a1|first/i,
    A2_uA: /a2|second/i,
    duration_ms: /duration|width/i,
    frequency_Hz: /freq/i,
    g_hrs_uS: /hrs|high/i,
    g_lrs_uS: /lrs|low/i,
    level: /level/i,
    sweep: /sweep/i,
  };

  for (const col of required) {
    const pat = patterns[col];
    if (pat) {
      const match = headers.find((h) => pat.test(h));
      mapping[col] = match || null;
    } else {
      mapping[col] = null;
    }
  }
  return mapping;
}

function TestCard({ test }: { test: TestType }) {
  const [expanded, setExpanded] = useState(false);
  const { uploadedTests, setUploadedTest, removeUploadedTest, setIsCurrentInput } = useAppStore();
  const uploaded = uploadedTests[test.id];
  const [mapping, setMapping] = useState<ColumnMapping>({});

  const handleFileLoaded = (dataset: Dataset) => {
    const autoMap = autoMapColumns(dataset.headers, [
      ...test.requiredColumns,
      ...test.optionalColumns,
    ]);
    setMapping(autoMap);

    // Detect if current column is present instead of conductance
    const hasCurrentCol = dataset.headers.some((h) => /current|i_?ua/i.test(h));
    const hasCondCol = dataset.headers.some((h) => /conductance|cond|g_?us/i.test(h));
    if (hasCurrentCol && !hasCondCol) {
      setIsCurrentInput(true);
    }

    setUploadedTest(test.id, {
      testId: test.id,
      dataset,
      columnMapping: autoMap,
    });
  };

  // Build preview chart data from first numeric column
  const previewChart = (() => {
    if (!uploaded) return null;
    const numCol = uploaded.dataset.columns.find((c) => c.values.length > 0);
    if (!numCol || numCol.values.length < 2) return null;
    const chartData = numCol.values.map((v, i) => ({ index: i + 1, [numCol.name]: v }));
    return { data: chartData, dataKey: numCol.name };
  })();

  return (
    <div
      className={`rounded-xl border transition-all ${
        test.critical
          ? 'border-green/20 bg-green/5'
          : 'border-amber/15 bg-amber/3'
      } ${uploaded ? 'ring-1 ring-accent/30' : ''}`}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left"
      >
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center ${
            test.critical ? 'bg-green/10 text-green' : 'bg-amber/10 text-amber'
          }`}
        >
          {iconMap[test.icon] || <Zap size={18} />}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-text">{test.name}</span>
            {test.critical && (
              <span className="text-[10px] px-1.5 py-0.5 bg-green/15 text-green rounded-full">
                ANN Required
              </span>
            )}
            {uploaded && (
              <span className="text-[10px] px-1.5 py-0.5 bg-accent/15 text-accent rounded-full">
                Loaded
              </span>
            )}
          </div>
          <p className="text-xs text-text-dim truncate">{test.description}</p>
        </div>
        {expanded ? (
          <ChevronDown size={16} className="text-text-dim" />
        ) : (
          <ChevronRight size={16} className="text-text-dim" />
        )}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          <div className="text-xs text-text-muted">
            <span className="font-medium">Required:</span>{' '}
            {test.requiredColumns.map((c) => (
              <code key={c} className="font-mono bg-surface-alt px-1 py-0.5 rounded mx-0.5">
                {c}
              </code>
            ))}
          </div>

          {test.id === 'epsc' && (
            <div className="p-3 bg-amber/5 border border-amber/20 rounded-lg text-xs text-amber leading-relaxed">
              If your current shifts to a new baseline without decay after the pulse, that indicates strong Long-Term Potentiation (LTP) with minimal Short-Term Plasticity — this is actually favorable for non-volatile memory applications.
            </div>
          )}

          <details className="text-xs text-text-dim">
            <summary className="cursor-pointer hover:text-text-muted">How to record</summary>
            <p className="mt-1 pl-3 border-l border-border">{test.recordingGuide}</p>
          </details>

          <FileUploader onFileLoaded={handleFileLoaded} />

          {uploaded && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-muted">
                  {uploaded.dataset.filename} — {uploaded.dataset.rows.length} rows
                </span>
                <button
                  onClick={() => removeUploadedTest(test.id)}
                  className="text-xs text-red hover:text-red/80"
                >
                  Remove
                </button>
              </div>

              {/* Column mapping */}
              <div className="grid grid-cols-2 gap-2">
                {test.requiredColumns.map((col) => (
                  <div key={col} className="flex items-center gap-2">
                    <span className="text-xs text-text-dim w-28 truncate font-mono">{col}:</span>
                    <select
                      value={mapping[col] || ''}
                      onChange={(e) => {
                        const newMap = { ...mapping, [col]: e.target.value || null };
                        setMapping(newMap);
                        setUploadedTest(test.id, { ...uploaded, columnMapping: newMap });
                      }}
                      className="flex-1 bg-surface-alt border border-border rounded px-2 py-1 text-xs text-text"
                    >
                      <option value="">— select —</option>
                      {uploaded.dataset.headers.map((h) => (
                        <option key={h} value={h}>
                          {h}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>

              <DataTable
                headers={uploaded.dataset.headers}
                rows={uploaded.dataset.rows}
                maxRows={5}
              />

              {/* Data preview chart */}
              {previewChart && (
                <AppLineChart
                  data={previewChart.data}
                  lines={[{ dataKey: previewChart.dataKey, color: '#4f8ff7', name: previewChart.dataKey }]}
                  xKey="index"
                  xLabel="Index"
                  yLabel={previewChart.dataKey}
                  heightOverride={200}
                  id={`preview-${test.id}`}
                />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function UploadPage() {
  const critical = TEST_TYPES.filter((t) => t.critical);
  const optional = TEST_TYPES.filter((t) => !t.critical);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-text mb-1">Upload Experimental Data</h2>
        <p className="text-sm text-text-muted">
          Upload your Keithley SMU data files. Green cards feed the ANN simulation.
        </p>
      </div>

      <div>
        <h3 className="text-sm font-medium text-green mb-3">Required for ANN Simulation</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {critical.map((t) => (
            <TestCard key={t.id} test={t} />
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-medium text-amber mb-3">Paper Figures Only</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {optional.map((t) => (
            <TestCard key={t.id} test={t} />
          ))}
        </div>
      </div>
    </div>
  );
}
