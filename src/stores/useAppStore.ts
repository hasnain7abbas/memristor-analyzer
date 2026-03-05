import { create } from 'zustand';
import type {
  TabId,
  SmoothingConfig,
  ExtractedParams,
  ANNConfig,
  ANNEpochResult,
  GraphStyle,
  UploadedTest,
} from '../types';

interface AppState {
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;

  uploadedTests: Record<string, UploadedTest>;
  setUploadedTest: (testId: string, test: UploadedTest) => void;
  removeUploadedTest: (testId: string) => void;

  smoothingConfig: SmoothingConfig;
  setSmoothingConfig: (config: Partial<SmoothingConfig>) => void;

  smoothedData: Record<string, number[]>;
  setSmoothedData: (key: string, data: number[]) => void;

  extractedParams: ExtractedParams | null;
  setExtractedParams: (params: ExtractedParams | null) => void;

  annConfig: ANNConfig;
  setANNConfig: (config: Partial<ANNConfig>) => void;

  annResults: ANNEpochResult[];
  setANNResults: (results: ANNEpochResult[]) => void;
  addANNResult: (result: ANNEpochResult) => void;

  isTraining: boolean;
  setIsTraining: (v: boolean) => void;

  graphStyle: GraphStyle;
  setGraphStyle: (style: Partial<GraphStyle>) => void;

  vRead: number;
  setVRead: (v: number) => void;

  isCurrentInput: boolean;
  setIsCurrentInput: (v: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeTab: 'upload',
  setActiveTab: (tab) => set({ activeTab: tab }),

  uploadedTests: {},
  setUploadedTest: (testId, test) =>
    set((s) => ({ uploadedTests: { ...s.uploadedTests, [testId]: test } })),
  removeUploadedTest: (testId) =>
    set((s) => {
      const copy = { ...s.uploadedTests };
      delete copy[testId];
      return { uploadedTests: copy };
    }),

  smoothingConfig: {
    method: 'savitzky_golay',
    windowSize: 5,
    polyOrder: 2,
    removeOutliers: false,
    outlierSigma: 2.5,
    strength: 1.0,
    enforceMonotonic: false,
    monotonicDirection: 'auto',
    bandwidth: 0.3,
  },
  setSmoothingConfig: (config) =>
    set((s) => ({ smoothingConfig: { ...s.smoothingConfig, ...config } })),

  smoothedData: {},
  setSmoothedData: (key, data) =>
    set((s) => ({ smoothedData: { ...s.smoothedData, [key]: data } })),

  extractedParams: null,
  setExtractedParams: (params) => set({ extractedParams: params }),

  annConfig: {
    modelType: 'mlp_1h',
    hiddenSize: 128,
    hiddenSize2: 64,
    epochs: 50,
    learningRate: 0.03,
    batchSize: 32,
  },
  setANNConfig: (config) =>
    set((s) => ({ annConfig: { ...s.annConfig, ...config } })),

  annResults: [],
  setANNResults: (results) => set({ annResults: results }),
  addANNResult: (result) =>
    set((s) => ({ annResults: [...s.annResults, result] })),

  isTraining: false,
  setIsTraining: (v) => set({ isTraining: v }),

  graphStyle: {
    width: 600,
    height: 450,
    dpi: 300,
    fontFamily: 'Times New Roman',
    axisFontSize: 12,
    titleFontSize: 14,
    tickFontSize: 11,
    lineWidth: 2,
    markerSize: 4,
    backgroundColor: '#ffffff',
    axisColor: '#000000',
    gridColor: '#cccccc',
    showGrid: true,
    gridOpacity: 0.3,
    showLegend: true,
    showBorder: true,
    borderColor: '#000000',
    borderWidth: 1,
  },
  setGraphStyle: (style) =>
    set((s) => ({ graphStyle: { ...s.graphStyle, ...style } })),

  vRead: 0.1,
  setVRead: (v) => set({ vRead: v }),

  isCurrentInput: false,
  setIsCurrentInput: (v) => set({ isCurrentInput: v }),
}));
