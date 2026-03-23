export interface DataColumn {
  name: string;
  values: number[];
}

export interface Dataset {
  filename: string;
  sheetName?: string;
  headers: string[];
  rows: Record<string, number | string>[];
  columns: DataColumn[];
}

export interface TestType {
  id: string;
  name: string;
  icon: string;
  critical: boolean;
  description: string;
  requiredColumns: string[];
  optionalColumns: string[];
  recordingGuide: string;
}

export interface SmoothingConfig {
  method: 'none' | 'moving_avg' | 'median' | 'savitzky_golay' | 'loess' | 'gaussian';
  windowSize: number;
  polyOrder: number;
  removeOutliers: boolean;
  outlierSigma: number;
  strength: number;
  enforceMonotonic: boolean;
  monotonicDirection: 'increasing' | 'decreasing' | 'auto';
  bandwidth: number;
}

export interface ExtractedParams {
  Gmin: number;
  Gmax: number;
  onOffRatio: number;
  dynamicRange: number;
  alphaP: number;
  alphaD: number;
  rSquaredP: number;
  rSquaredD: number;
  ccvPercent: number;
  writeNoise: number;
  numLevelsP: number;
  numLevelsD: number;
  potentiationRaw: number[];
  potentiationSmoothed: number[];
  potentiationFitted: number[];
  depressionRaw: number[];
  depressionSmoothed: number[];
  depressionFitted: number[];
  deltaG: { G: number; dG: number }[];
  memoryWindow: number;
  programmingMargin: number;
  asymmetryIndex: number;
  switchingUniformity: number;
}

export type ANNModelType = 'perceptron' | 'mlp_1h' | 'mlp_2h' | 'lenet5' | 'cnn_simple';

export interface ANNConfig {
  modelType: ANNModelType;
  hiddenSize: number;
  hiddenSize2: number;
  epochs: number;
  learningRate: number;
  batchSize: number;
}

export interface ANNEpochResult {
  epoch: number;
  idealAccuracy: number;
  memristorAccuracy: number;
  idealLoss: number;
  memristorLoss: number;
}

export interface GraphStyle {
  width: number;
  height: number;
  dpi: number;
  fontFamily: string;
  axisFontSize: number;
  titleFontSize: number;
  tickFontSize: number;
  lineWidth: number;
  markerSize: number;
  backgroundColor: string;
  axisColor: string;
  gridColor: string;
  showGrid: boolean;
  gridOpacity: number;
  showLegend: boolean;
  showBorder: boolean;
  borderColor: string;
  borderWidth: number;
}

export type PlotType = 'line' | 'scatter' | 'line_scatter' | 'step';

export interface ChartLocalSettings {
  xLabel: string;
  yLabel: string;
  plotType: PlotType;
  caption: string;
}

export type TabId = 'upload' | 'smoothing' | 'parameters' | 'ann' | 'export';

export interface ColumnMapping {
  [expectedColumn: string]: string | null;
}

export interface UploadedTest {
  testId: string;
  dataset: Dataset;
  columnMapping: ColumnMapping;
  vRead?: number;
}

export type FrameworkType = 'pytorch' | 'crosssim' | 'neurosim' | 'memtorch';

export interface FormulaSection {
  subtitle: string;
  content: string;
  latex?: string;
}

export interface FormulaDefinition {
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
