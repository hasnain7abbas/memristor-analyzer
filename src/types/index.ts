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
  method: 'none' | 'moving_avg' | 'median' | 'savitzky_golay';
  windowSize: number;
  polyOrder: number;
  removeOutliers: boolean;
  outlierSigma: number;
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
}

export interface ANNConfig {
  hiddenSize: number;
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
  lineWidth: number;
  markerSize: number;
  backgroundColor: string;
  axisColor: string;
  showGrid: boolean;
  gridOpacity: number;
  showLegend: boolean;
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
