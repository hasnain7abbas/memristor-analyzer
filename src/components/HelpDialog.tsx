import { X } from 'lucide-react';
import { Tex } from './Tex';

interface HelpDialogProps {
  onClose: () => void;
}

/* ──────────────────── Parameter Table Row ──────────────────── */
function ParamRow({ symbol, name, unit, desc }: { symbol: string; name: string; unit: string; desc: string }) {
  return (
    <tr className="border-b border-border/50">
      <td className="py-2 pr-3 font-mono text-accent"><Tex math={symbol} /></td>
      <td className="py-2 pr-3 text-text font-medium">{name}</td>
      <td className="py-2 pr-3 text-text-muted font-mono text-xs">{unit}</td>
      <td className="py-2 text-text-muted text-sm">{desc}</td>
    </tr>
  );
}

/* ──────────────────── Quality Table ──────────────────── */
function QualityRow({ range, label, color }: { range: string; label: string; color: string }) {
  return (
    <tr className="border-b border-border/50">
      <td className="py-1.5 pr-4 font-mono text-sm text-text">{range}</td>
      <td className={`py-1.5 font-medium text-sm ${color}`}>{label}</td>
    </tr>
  );
}

/* ──────────────────── Section Wrapper ──────────────────── */
function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-4">
      <h2 className="text-lg font-semibold text-accent mb-4 border-b border-accent/20 pb-2">{title}</h2>
      <div className="space-y-4">{children}</div>
    </section>
  );
}

function SubSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="ml-1 space-y-2">
      <h3 className="text-sm font-semibold text-purple">{title}</h3>
      <div className="space-y-2 text-sm text-text-muted leading-relaxed">{children}</div>
    </div>
  );
}

function FormulaBlock({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-bg/80 border border-border/50 rounded-lg p-4 my-3 text-center overflow-x-auto">
      {children}
    </div>
  );
}

/* ──────────────────── MAIN HELP DIALOG ──────────────────── */
export function HelpDialog({ onClose }: HelpDialogProps) {
  const sections = [
    { id: 'overview', label: 'Overview' },
    { id: 'conductance', label: '1. Conductance' },
    { id: 'window', label: '2. Conductance Window' },
    { id: 'nonlinearity', label: '3. Non-Linearity (α)' },
    { id: 'noise', label: '4. Write Noise & CCV' },
    { id: 'delta-g', label: '5. ΔG vs G Analysis' },
    { id: 'ppf', label: '6. PPF Index' },
    { id: 'ann', label: '7. ANN Simulation' },
    { id: 'acquisition', label: '8. Data Acquisition Guide' },
  ];

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center bg-black/60 backdrop-blur-sm overflow-y-auto py-8">
      <div className="w-full max-w-4xl bg-surface border border-border rounded-2xl shadow-2xl mx-4 mb-8">
        {/* Header */}
        <div className="sticky top-0 z-10 bg-surface border-b border-border rounded-t-2xl px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-text">Formula Reference &amp; Help</h1>
          <button onClick={onClose} className="p-2 hover:bg-surface-alt rounded-lg transition-colors">
            <X size={20} className="text-text-muted" />
          </button>
        </div>

        {/* Navigation */}
        <div className="sticky top-[65px] z-10 bg-surface/95 backdrop-blur border-b border-border/50 px-6 py-2 flex flex-wrap gap-1">
          {sections.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className="px-2.5 py-1 rounded text-xs font-medium text-text-muted hover:text-accent hover:bg-accent/10 transition-colors"
            >
              {s.label}
            </a>
          ))}
        </div>

        {/* Content */}
        <div className="px-6 py-6 space-y-10">

          {/* ─── OVERVIEW ─── */}
          <Section id="overview" title="Overview">
            <p className="text-sm text-text-muted leading-relaxed">
              This tool characterizes memristor devices for neuromorphic computing applications.
              It extracts key device parameters from experimental potentiation/depression (P/D) cycling data
              and evaluates how those parameters affect Artificial Neural Network (ANN) performance.
            </p>
            <p className="text-sm text-text-muted leading-relaxed">
              The workflow is: <strong className="text-text">Upload</strong> raw measurement data
              → <strong className="text-text">Smooth</strong> noisy curves
              → <strong className="text-text">Extract</strong> device parameters (α, σ<sub>w</sub>, G<sub>min</sub>/G<sub>max</sub>)
              → <strong className="text-text">Simulate</strong> ANN training with memristor non-idealities
              → <strong className="text-text">Export</strong> publication-quality figures and Python scripts.
            </p>
          </Section>

          {/* ─── 1. CONDUCTANCE ─── */}
          <Section id="conductance" title="1. Conductance (G)">
            <SubSection title="Core Formula">
              <FormulaBlock>
                <Tex math="G = \frac{I_{\text{read}}}{V_{\text{read}}}" display />
              </FormulaBlock>
              <p>
                Conductance is the reciprocal of resistance. When a small read voltage <Tex math="V_{\text{read}}" /> is applied
                across the memristor, the resulting current <Tex math="I_{\text{read}}" /> is measured. Dividing current by
                voltage yields conductance.
              </p>
            </SubSection>

            <SubSection title="Parameters">
              <table className="w-full text-left">
                <tbody>
                  <ParamRow symbol="G" name="Conductance" unit="µS" desc="Ease of current flow through the memristive device. Higher G = lower resistance = thicker/more connected filament." />
                  <ParamRow symbol="I_{\text{read}}" name="Read Current" unit="µA" desc="Current measured by the sourcemeter during the read phase at fixed V_read." />
                  <ParamRow symbol="V_{\text{read}}" name="Read Voltage" unit="V" desc="Small non-disturbing voltage applied during read (typically 0.05–0.2 V). Must be below the SET/RESET threshold." />
                </tbody>
              </table>
            </SubSection>

            <SubSection title="Derivation from Ohm's Law">
              <FormulaBlock>
                <Tex math="V = IR \;\;\Rightarrow\;\; I = GV \;\;\Rightarrow\;\; G = \frac{I}{V} = \frac{1}{R}" display />
              </FormulaBlock>
              <p>
                Ohm's law states <Tex math="V = IR" />. Conductance <Tex math="G = 1/R" /> is the inverse of resistance.
                For a memristor, conductance represents the synaptic weight in neuromorphic computing.
              </p>
            </SubSection>

            <SubSection title="Unit Conversions">
              <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-bg/40 rounded-lg p-3">
                <div>1 A = 10⁶ µA = 10³ mA = 10⁹ nA</div>
                <div>1 S = 10⁶ µS = 10³ mS = 10⁹ nS</div>
              </div>
            </SubSection>

            <SubSection title="How to Acquire">
              <ol className="list-decimal ml-5 space-y-1.5 text-sm text-text-muted">
                <li>Connect your memristor device to a Keithley 2400/2450 or similar sourcemeter.</li>
                <li>Choose a read voltage (e.g., <Tex math="V_{\text{read}} = 0.1\,\text{V}" />) below the SET/RESET threshold.</li>
                <li>After each programming pulse, apply the read voltage and record the current.</li>
                <li>Compute <Tex math="G = I / V_{\text{read}}" /> for each measurement point.</li>
                <li>If your instrument records current directly, this tool converts to conductance automatically — just enter <Tex math="V_{\text{read}}" />.</li>
              </ol>
            </SubSection>

            <SubSection title="Physical Meaning">
              <p>
                In filamentary memristors (e.g., HfO₂, TiO₂, TaOₓ), conductance is proportional to the cross-sectional
                area of the conductive filament formed by oxygen vacancy migration. A thick, well-connected filament
                gives high conductance (LRS), while a thin or ruptured filament gives low conductance (HRS).
              </p>
            </SubSection>
          </Section>

          {/* ─── 2. CONDUCTANCE WINDOW ─── */}
          <Section id="window" title="2. Conductance Window (G_min, G_max, On/Off Ratio)">
            <SubSection title="Core Formulas">
              <FormulaBlock>
                <div className="space-y-3">
                  <Tex math="G_{\min} = \min(\text{potentiation curve}), \quad G_{\max} = \max(\text{potentiation curve})" display />
                  <Tex math="\text{On/Off Ratio} = \frac{G_{\max}}{G_{\min}}, \quad \Delta G = G_{\max} - G_{\min}" display />
                </div>
              </FormulaBlock>
            </SubSection>

            <SubSection title="Parameters">
              <table className="w-full text-left">
                <tbody>
                  <ParamRow symbol="G_{\min}" name="Minimum Conductance" unit="µS" desc="High Resistance State (HRS). The conductance when the filament is thinnest/ruptured. Corresponds to the weakest synaptic weight." />
                  <ParamRow symbol="G_{\max}" name="Maximum Conductance" unit="µS" desc="Low Resistance State (LRS). The conductance when the filament is thickest/fully formed. Corresponds to the strongest synaptic weight." />
                  <ParamRow symbol="\text{On/Off}" name="On/Off Ratio" unit="—" desc="Signal-to-noise margin for weight storage. Higher = better distinguishability between states. Typical: 2–100×." />
                  <ParamRow symbol="\Delta G" name="Dynamic Range" unit="µS" desc="Total conductance swing available for analog weight programming." />
                </tbody>
              </table>
            </SubSection>

            <SubSection title="Derived Metrics">
              <FormulaBlock>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-text-dim mb-1">Memory Window (dB):</p>
                    <Tex math="\text{MW} = 20 \cdot \log_{10}\!\left(\frac{G_{\max}}{G_{\min}}\right)" display />
                  </div>
                  <div>
                    <p className="text-xs text-text-dim mb-1">Programming Margin:</p>
                    <Tex math="\text{PM} = \frac{G_{\max} - G_{\min}}{G_{\max} + G_{\min}} \times 100\%" display />
                  </div>
                  <div>
                    <p className="text-xs text-text-dim mb-1">Number of Distinguishable Levels:</p>
                    <Tex math="N = \frac{\Delta G}{\bar{\Delta G}_{\text{step}} + 2\,\sigma_{\text{step}}}" display />
                  </div>
                </div>
              </FormulaBlock>
              <p><strong className="text-text">Memory Window</strong>: logarithmic measure of the On/Off ratio. 20 dB = 10:1 ratio, 40 dB = 100:1.</p>
              <p><strong className="text-text">Programming Margin</strong>: PM &gt; 50% indicates good separation between HRS and LRS. PM &lt; 20% risks read errors.</p>
              <p><strong className="text-text">Distinguishable Levels</strong>: the number of reliably programmable analog states, accounting for noise. More levels = more bits per synapse.</p>
            </SubSection>

            <SubSection title="How to Acquire">
              <ol className="list-decimal ml-5 space-y-1">
                <li>Run a full potentiation sweep (repeated SET pulses) recording conductance after each pulse.</li>
                <li>The minimum value in the sweep is <Tex math="G_{\min}" />; the maximum is <Tex math="G_{\max}" />.</li>
                <li>For multi-cycle data, take the average <Tex math="G_{\min}" /> and <Tex math="G_{\max}" /> across cycles.</li>
              </ol>
            </SubSection>

            <SubSection title="Quality Thresholds">
              <table className="text-left">
                <tbody>
                  <QualityRow range="On/Off > 10" label="Excellent" color="text-green" />
                  <QualityRow range="On/Off 3–10" label="Acceptable" color="text-amber" />
                  <QualityRow range="On/Off < 3" label="Poor — limited analog levels" color="text-red" />
                </tbody>
              </table>
            </SubSection>
          </Section>

          {/* ─── 3. NON-LINEARITY ─── */}
          <Section id="nonlinearity" title="3. Non-Linearity α — The Key Parameter">
            <SubSection title="Core Formula">
              <FormulaBlock>
                <Tex math="G(n) = G_{\text{start}} + \bigl(G_{\text{end}} - G_{\text{start}}\bigr) \cdot \frac{1 - e^{-\alpha \cdot n/N}}{1 - e^{-\alpha}}" display />
              </FormulaBlock>
              <p>
                This is the non-linear conductance update model. It describes how conductance changes as a function
                of pulse number <Tex math="n" /> during potentiation or depression. The parameter <Tex math="\alpha" />{' '}
                controls the degree of non-linearity.
              </p>
            </SubSection>

            <SubSection title="Parameters">
              <table className="w-full text-left">
                <tbody>
                  <ParamRow symbol="G(n)" name="Conductance at Pulse n" unit="µS" desc="The conductance state after applying n programming pulses." />
                  <ParamRow symbol="G_{\text{start}}" name="Starting Conductance" unit="µS" desc="Conductance at the beginning of the sweep (n=0). For potentiation: G_min. For depression: G_max." />
                  <ParamRow symbol="G_{\text{end}}" name="Ending Conductance" unit="µS" desc="Conductance at the end of the sweep (n=N). For potentiation: G_max. For depression: G_min." />
                  <ParamRow symbol="\alpha" name="Non-Linearity Parameter" unit="—" desc="Controls curvature. α→0: perfectly linear. α>0: compressive (saturates early). Separate α_P (potentiation) and α_D (depression)." />
                  <ParamRow symbol="n" name="Pulse Number" unit="—" desc="Current pulse index, from 0 to N." />
                  <ParamRow symbol="N" name="Total Pulses" unit="—" desc="Total number of programming pulses in the sweep." />
                </tbody>
              </table>
            </SubSection>

            <SubSection title="Physical Derivation">
              <FormulaBlock>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-text-dim mb-1">Kinetic equation for filament growth:</p>
                    <Tex math="\frac{dG}{dn} = \beta \cdot (G_{\text{end}} - G)^{\gamma}" display />
                  </div>
                  <div>
                    <p className="text-xs text-text-dim mb-1">Solving this ODE with boundary conditions gives:</p>
                    <Tex math="G(n) = G_{\text{start}} + (G_{\text{end}} - G_{\text{start}}) \cdot \frac{1 - e^{-\alpha \cdot n/N}}{1 - e^{-\alpha}}" display />
                  </div>
                </div>
              </FormulaBlock>
              <p>
                The non-linearity arises from the electric field distribution within the switching layer.
                As the conductive filament grows (potentiation), the electric field concentrates at the filament tip,
                causing rapid initial growth that gradually saturates as the gap narrows. During depression (filament
                dissolution), the reverse process occurs with its own characteristic non-linearity.
              </p>
            </SubSection>

            <SubSection title="How α Is Fitted">
              <ol className="list-decimal ml-5 space-y-1">
                <li>The smoothed P/D data is extracted from your measurement.</li>
                <li>A coarse grid search tests <Tex math="\alpha" /> from 0.01 to 12.0 in steps of 0.05.</li>
                <li>For each candidate <Tex math="\alpha" />, the model curve <Tex math="G(n)" /> is computed and the sum of squared errors (SSE) against the data is calculated.</li>
                <li>A fine grid search refines ±0.3 around the best coarse result in steps of 0.001.</li>
                <li>The goodness of fit is reported as <Tex math="R^2 = 1 - \text{SSE}/\text{SST}" />.</li>
              </ol>
            </SubSection>

            <SubSection title="Separate α for Potentiation and Depression">
              <FormulaBlock>
                <div className="space-y-3">
                  <Tex math="\alpha_P \text{ (Potentiation):  fits the SET / LTP sweep}" display />
                  <Tex math="\alpha_D \text{ (Depression):  fits the RESET / LTD sweep}" display />
                  <div>
                    <p className="text-xs text-text-dim mb-1">Asymmetry Index:</p>
                    <Tex math="\text{AI} = \frac{|\alpha_P - \alpha_D|}{\max(\alpha_P,\, \alpha_D)}" display />
                  </div>
                </div>
              </FormulaBlock>
              <p>
                AI = 0 means perfectly symmetric updates (ideal). AI &gt; 0.5 means highly asymmetric — the device
                potentiates and depresses very differently, causing weight drift during ANN training.
              </p>
            </SubSection>

            <SubSection title="Impact on ANN Accuracy">
              <table className="text-left">
                <tbody>
                  <QualityRow range="α < 1" label="< 5% accuracy drop (excellent)" color="text-green" />
                  <QualityRow range="α = 1–2" label="5–15% accuracy drop (acceptable)" color="text-green" />
                  <QualityRow range="α = 2–4" label="15–30% accuracy drop (degraded)" color="text-amber" />
                  <QualityRow range="α > 4" label="> 30% accuracy drop (poor — needs compensation)" color="text-red" />
                </tbody>
              </table>
            </SubSection>

            <SubSection title="Reference">
              <p className="text-xs text-text-dim italic">
                G. W. Burr et al., "Neuromorphic computing using non-volatile memory," <em>Advances in Physics: X</em>, vol. 2, no. 1, 2017.
              </p>
            </SubSection>
          </Section>

          {/* ─── 4. WRITE NOISE & CCV ─── */}
          <Section id="noise" title="4. Cycle-to-Cycle Variation (CCV) &amp; Write Noise (σ_w)">
            <SubSection title="Core Formulas">
              <FormulaBlock>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-text-dim mb-1">Conductance step between consecutive pulses:</p>
                    <Tex math="\Delta G_i = G_{i+1} - G_i" display />
                  </div>
                  <div>
                    <p className="text-xs text-text-dim mb-1">Cycle-to-Cycle Variation (coefficient of variation):</p>
                    <Tex math="\text{CCV}\% = \frac{\sigma(|\Delta G|)}{\mu(|\Delta G|)} \times 100\%" display />
                  </div>
                  <div>
                    <p className="text-xs text-text-dim mb-1">Normalized Write Noise:</p>
                    <Tex math="\sigma_w = \frac{\sigma(\Delta G)}{G_{\max} - G_{\min}}" display />
                  </div>
                </div>
              </FormulaBlock>
            </SubSection>

            <SubSection title="Parameters">
              <table className="w-full text-left">
                <tbody>
                  <ParamRow symbol="\Delta G_i" name="Conductance Step" unit="µS" desc="Change in conductance due to the i-th programming pulse. Ideally constant; in practice varies stochastically." />
                  <ParamRow symbol="\sigma(|\Delta G|)" name="Std Dev of Steps" unit="µS" desc="Standard deviation of the absolute conductance step sizes. Measures pulse-to-pulse reproducibility." />
                  <ParamRow symbol="\mu(|\Delta G|)" name="Mean Step Size" unit="µS" desc="Average absolute conductance change per pulse. The 'expected' update magnitude." />
                  <ParamRow symbol="\text{CCV}\%" name="Cycle-to-Cycle Variation" unit="%" desc="Relative variability of weight updates. Lower = more reproducible. Reported as a percentage." />
                  <ParamRow symbol="\sigma_w" name="Normalized Write Noise" unit="—" desc="Write noise normalized to the full conductance range. Dimensionless. Used directly in ANN simulation." />
                </tbody>
              </table>
            </SubSection>

            <SubSection title="Statistical Origin">
              <p>
                Write noise originates from the stochastic nature of ion migration and filament formation/dissolution.
                Each write pulse causes a slightly different atomic rearrangement, leading to conductance variations.
                The noise follows approximately <Tex math="\Delta G \sim \mathcal{N}(\mu_{\Delta G},\, \sigma_{\Delta G}^2)" />.
              </p>
              <FormulaBlock>
                <div className="space-y-2">
                  <p className="text-xs text-text-dim">Device area scaling:</p>
                  <Tex math="\sigma_w \propto \frac{1}{\sqrt{A}}" display />
                </div>
              </FormulaBlock>
              <p>Smaller devices have more stochastic behavior (fewer atoms involved in switching).</p>
              <div className="text-xs font-mono bg-bg/40 rounded-lg p-3 space-y-1">
                <p>50 nm device → σ_w ≈ 0.01–0.05</p>
                <p>1 µm device  → σ_w ≈ 0.001–0.01</p>
              </div>
            </SubSection>

            <SubSection title="Multi-Cycle Formula">
              <FormulaBlock>
                <Tex math="\text{CCV}_{\text{total}} = \sqrt{\text{CCV}_{\text{intra}}^2 + \text{CCV}_{\text{inter}}^2}" display />
              </FormulaBlock>
              <p>
                <strong className="text-text">CCV<sub>intra</sub></strong>: variation within a single potentiation/depression cycle.{' '}
                <strong className="text-text">CCV<sub>inter</sub></strong>: variation between different cycles.
                This tool computes CCV from all available data; if multi-cycle data is uploaded, both components are captured.
              </p>
            </SubSection>

            <SubSection title="How to Acquire">
              <ol className="list-decimal ml-5 space-y-1">
                <li>Run multiple potentiation/depression cycles (ideally 5+ cycles).</li>
                <li>Record conductance after every single pulse.</li>
                <li>Upload all cycles — the tool automatically computes ΔG between consecutive pulses.</li>
                <li>If a "cycle" column is present in your data, the tool separates intra- and inter-cycle variation.</li>
              </ol>
            </SubSection>

            <SubSection title="Quality Thresholds">
              <table className="text-left">
                <tbody>
                  <QualityRow range="σ_w < 0.01" label="Excellent (low noise)" color="text-green" />
                  <QualityRow range="σ_w = 0.01–0.05" label="Typical for nanoscale devices" color="text-amber" />
                  <QualityRow range="σ_w > 0.05" label="High noise — significant ANN degradation" color="text-red" />
                </tbody>
              </table>
            </SubSection>
          </Section>

          {/* ─── 5. DELTA-G ─── */}
          <Section id="delta-g" title="5. ΔG vs G Scatter Analysis">
            <SubSection title="What Is Plotted">
              <FormulaBlock>
                <Tex math="\text{For each pulse } i: \quad x = G_i, \quad y = G_{i+1} - G_i" display />
              </FormulaBlock>
              <p>
                This scatter plot reveals the <em>state-dependent</em> switching behavior. Each point represents one
                programming pulse: the x-coordinate is the current conductance state, and the y-coordinate is how much
                that state changed.
              </p>
            </SubSection>

            <SubSection title="State-Dependent Model">
              <FormulaBlock>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-text-dim mb-1">Potentiation:</p>
                    <Tex math="\Delta G(G) = \alpha_{\text{eff}} \cdot (G_{\max} - G)" display />
                  </div>
                  <div>
                    <p className="text-xs text-text-dim mb-1">Depression:</p>
                    <Tex math="\Delta G(G) = -\alpha_{\text{eff}} \cdot (G - G_{\min})" display />
                  </div>
                </div>
              </FormulaBlock>
              <p>
                A <strong className="text-text">negative slope</strong> in the potentiation ΔG vs G means the device updates
                less as it approaches <Tex math="G_{\max}" /> (saturation). A <strong className="text-text">flat line</strong>{' '}
                (zero slope) means perfectly uniform updates — the ideal case for ANN training.
              </p>
            </SubSection>

            <SubSection title="Switching Uniformity Index (SUI)">
              <FormulaBlock>
                <Tex math="\text{SUI} = 1 - \bigl|\text{slope of linear regression of } \Delta G \text{ vs } G\bigr|" display />
              </FormulaBlock>
              <table className="text-left">
                <tbody>
                  <QualityRow range="SUI ≈ 1" label="Perfectly uniform (linear device)" color="text-green" />
                  <QualityRow range="SUI > 0.7" label="Good uniformity" color="text-green" />
                  <QualityRow range="SUI 0.5–0.7" label="Moderate non-uniformity" color="text-amber" />
                  <QualityRow range="SUI < 0.5" label="Highly non-uniform" color="text-red" />
                </tbody>
              </table>
            </SubSection>
          </Section>

          {/* ─── 6. PPF ─── */}
          <Section id="ppf" title="6. Paired-Pulse Facilitation (PPF) Index">
            <SubSection title="Core Formula">
              <FormulaBlock>
                <Tex math="\text{PPF Index} = \frac{A_2 - A_1}{A_1} \times 100\%" display />
              </FormulaBlock>
              <p>
                <Tex math="A_1" /> is the response (current or conductance change) to the first pulse,
                and <Tex math="A_2" /> is the response to the second pulse delivered shortly after.
                A positive PPF means the second pulse is more effective — the device shows short-term synaptic plasticity.
              </p>
            </SubSection>

            <SubSection title="Double Exponential Decay Model">
              <FormulaBlock>
                <Tex math="\text{PPF}(\Delta t) = 1 + C_1 \, e^{-\Delta t / \tau_1} + C_2 \, e^{-\Delta t / \tau_2}" display />
              </FormulaBlock>
              <table className="w-full text-left">
                <tbody>
                  <ParamRow symbol="C_1,\, C_2" name="Amplitude Coefficients" unit="—" desc="Weights of the fast and slow decay components." />
                  <ParamRow symbol="\tau_1" name="Fast Time Constant" unit="ms" desc="50–200 ms. Corresponds to Ca²⁺ diffusion in biological synapses; ion redistribution in memristors." />
                  <ParamRow symbol="\tau_2" name="Slow Time Constant" unit="s" desc="1–10 s. Structural relaxation / thermally-assisted processes." />
                  <ParamRow symbol="\Delta t" name="Inter-Pulse Interval" unit="ms" desc="Time between the two pulses. PPF decreases as Δt increases." />
                </tbody>
              </table>
            </SubSection>

            <SubSection title="Biological Comparison">
              <div className="text-xs font-mono bg-bg/40 rounded-lg p-3 space-y-1">
                <p>Hippocampal synapse: τ₁ ≈ 50 ms, τ₂ ≈ 500 ms</p>
                <p>Memristor (typical):  τ₁ ≈ 100–500 ms, τ₂ ≈ 2–20 s</p>
              </div>
              <p className="text-sm text-text-muted">
                Closer to biological timescales = better suitability for neuromorphic temporal processing tasks
                (e.g., speech recognition, gesture recognition).
              </p>
            </SubSection>

            <SubSection title="Reference">
              <p className="text-xs text-text-dim italic">
                R. Zucker &amp; W. Regehr, "Short-term synaptic plasticity," <em>Annu. Rev. Physiol.</em>, vol. 64, pp. 355–405, 2002.
              </p>
            </SubSection>
          </Section>

          {/* ─── 7. ANN SIMULATION ─── */}
          <Section id="ann" title="7. ANN / MNIST Simulation">
            <SubSection title="Copy-and-Degrade Methodology">
              <p>
                The simulator trains an <strong className="text-text">ideal</strong> neural network on synthetic MNIST digits,
                then evaluates a <strong className="text-text">memristor</strong> version by copying the ideal weights and
                applying device non-idealities:
              </p>
              <ol className="list-decimal ml-5 space-y-1">
                <li>Train the ideal network for one epoch using standard backpropagation (SGD with momentum).</li>
                <li>Clone all weights from the ideal network to the memristor network.</li>
                <li>Apply non-linear weight remapping using your <Tex math="\alpha_P" /> and <Tex math="\alpha_D" /> values.</li>
                <li>Apply conductance quantization (<Tex math="N" /> levels) and write noise (<Tex math="\sigma_w" />).</li>
                <li>Evaluate both networks on the test set. The gap shows the accuracy drop due to device non-idealities.</li>
              </ol>
            </SubSection>

            <SubSection title="Non-Linear Weight Remapping">
              <FormulaBlock>
                <Tex math="w_{\text{mapped}} = \frac{1 - e^{-\alpha \cdot w_{\text{norm}}}}{1 - e^{-\alpha}}" display />
              </FormulaBlock>
              <p>
                Each ideal weight is normalized to [0, 1], passed through the non-linear mapping curve
                (same as the device's P/D characteristic), then denormalized. Weights in the upper half use{' '}
                <Tex math="\alpha_P" />, lower half use <Tex math="\alpha_D" />.
              </p>
            </SubSection>

            <SubSection title="Quantization and Noise Injection">
              <FormulaBlock>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-text-dim mb-1">Quantize to N levels:</p>
                    <Tex math="w_q = \frac{\text{round}(w_{\text{norm}} \cdot (N-1))}{N-1}" display />
                  </div>
                  <div>
                    <p className="text-xs text-text-dim mb-1">Add Gaussian write noise:</p>
                    <Tex math="w_{\text{final}} = w_q + \mathcal{N}\!\left(0,\; \frac{\sigma_w}{\sqrt{N}}\right)" display />
                  </div>
                </div>
              </FormulaBlock>
            </SubSection>

            <SubSection title="Training Details">
              <ul className="list-disc ml-5 space-y-1">
                <li><strong className="text-text">Optimizer</strong>: SGD with momentum (0.9) and cosine annealing learning rate schedule.</li>
                <li><strong className="text-text">Loss</strong>: Cross-entropy <Tex math="\mathcal{L} = -\log(p_{\text{target}})" /></li>
                <li><strong className="text-text">Activations</strong>: ReLU (hidden layers), Softmax (output layer).</li>
                <li><strong className="text-text">Initialization</strong>: He initialization <Tex math="\sigma = \sqrt{2/\text{fan\_in}}" /></li>
                <li><strong className="text-text">Batch training</strong>: Gradients are accumulated over each mini-batch and averaged before the weight update.</li>
                <li><strong className="text-text">Noise averaging</strong>: Memristor accuracy is averaged over 3 independent noise realizations per epoch for stability.</li>
                <li><strong className="text-text">Data</strong>: 5,000 synthetic MNIST training images, 1,000 test images (in-app). Python scripts use real MNIST (60,000 images).</li>
              </ul>
            </SubSection>

            <SubSection title="Interpreting the Curves">
              <p>
                <strong className="text-accent">Blue (Ideal)</strong>: Standard ANN with perfect weight storage — your upper bound.
                Should rise smoothly and converge.
              </p>
              <p>
                <strong className="text-red">Red (Memristor)</strong>: ANN with your device's non-idealities applied.
                Follows the ideal curve at a lower level. The gap is the <em>accuracy drop</em>.
              </p>
              <p>
                A <strong className="text-text">small, stable gap</strong> means your device is well-suited for neuromorphic computing.
                A <strong className="text-text">large or growing gap</strong> means the non-idealities are too severe — consider
                device engineering (lower α, more levels) or algorithm-level compensation.
              </p>
            </SubSection>
          </Section>

          {/* ─── 8. DATA ACQUISITION ─── */}
          <Section id="acquisition" title="8. Data Acquisition Guide">
            <SubSection title="Required Equipment">
              <ul className="list-disc ml-5 space-y-1">
                <li>Sourcemeter (Keithley 2400, 2450, 2600 series, or equivalent)</li>
                <li>Probe station with tungsten or BeCu probe tips</li>
                <li>Memristor device with accessible top and bottom electrodes</li>
                <li>PC with GPIB/USB instrument control (optional: LabVIEW, Python pyvisa)</li>
              </ul>
            </SubSection>

            <SubSection title="Measurement Protocol">
              <ol className="list-decimal ml-5 space-y-2">
                <li>
                  <strong className="text-text">Electroforming</strong> (if needed): Apply a gradually increasing voltage sweep
                  until initial filament formation. This is a one-time step.
                </li>
                <li>
                  <strong className="text-text">P/D Cycling</strong>: Alternate between potentiation (repeated SET pulses)
                  and depression (repeated RESET pulses). For each phase:
                  <ul className="list-disc ml-5 mt-1">
                    <li>Apply N identical programming pulses (e.g., N = 50–200).</li>
                    <li>After each pulse, apply a read voltage and record the current.</li>
                    <li>Repeat for 5–10 complete P/D cycles for statistical reliability.</li>
                  </ul>
                </li>
                <li>
                  <strong className="text-text">Data Format</strong>: Save as CSV or Excel with columns:
                  <div className="text-xs font-mono bg-bg/40 rounded-lg p-3 mt-1 space-y-0.5">
                    <p>pulse_number, conductance_uS   (or current_uA)</p>
                    <p>Optional: type (P/D), cycle (1,2,3...), voltage</p>
                  </div>
                </li>
              </ol>
            </SubSection>

            <SubSection title="Tips for Best Results">
              <ul className="list-disc ml-5 space-y-1">
                <li>Use the same read voltage for all measurements (consistency is key).</li>
                <li>Allow sufficient settling time between write and read operations.</li>
                <li>Record at least 50 pulses per phase for reliable α fitting.</li>
                <li>Multiple P/D cycles enable CCV computation — 5 cycles minimum recommended.</li>
                <li>If recording current, enter the exact V_read value in the Parameters tab for conversion.</li>
              </ul>
            </SubSection>
          </Section>

        </div>

        {/* Footer */}
        <div className="border-t border-border px-6 py-4 text-xs text-text-dim text-center">
          Memristor Neural Analyzer — Formula Reference v1.0
        </div>
      </div>
    </div>
  );
}
