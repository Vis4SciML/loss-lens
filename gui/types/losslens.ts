export interface LocalMetric {
  [key: string]: number
}

export interface ModeNode {
  modeId: string
  modelId: string
  name: string
  localMetric: LocalMetric
  localFlatness: number[]
  x: number
  y: number
}

export interface ModeConnectivityLink {
  modePairId: string
  source: {
    modeId: string
    x: number
    y: number
  }
  target: {
    modeId: string
    x: number
    y: number
  }
  type: string
  weight: number
  distance: number
}

export interface SemiGlobalLocalStructure {
  nodes: ModeNode[]
  links: ModeConnectivityLink[]
  modelList: string[]
}

export interface SystemConfig {
  caseStudyList: string[]
  caseStudyLabels: {
    [key: string]: string
  }
  selectedCaseStudy: string | null
}

export interface ModelMetaData {
  modelId: string
  modelName: string
  modelDescription: string
  modelDataset: string
  datasetId: string
  modelDatasetDescription: string
  numberOfModes: number
}

export interface GlobalInfo {
  lossBounds: {
    upperBound: number
    lowerBound: number
  }
}

export interface LossLandscape {
  caseId: string
  modelId: string
  modeId: string
  grid: number[][]
  upperBound: number
  lowerBound: number
}

export interface PersistenceBarcode {
  caseId: string
  modelId: string
  modeId: string
  edges: Array<{
    y1: number
    y0: number
    x: number
  }>
}

export interface MergeTreeData {
  modeId: string
  nodes: Array<{
    id: number
    x: number
    y: number
  }>
  edges: Array<{
    sourceX: number
    sourceY: number
    targetX: number
    targetY: number
  }>
}

export interface LayerSimilarityData {
  caseId: string
  modePairId: string
  grid: number[][]
  xLabels: string[]
  yLabels: string[]
  upperBound: number
  lowerBound: number
}

export interface ConfusionMaterixBarData {
  caseId: string
  modesId: string[]
  modePairId: string
  classesName: string[]
  grid: {
    tp: [number, number]
    fp: [number, number]
    fn: [number, number]
    tn: [number, number]
  }[]
}

export interface ModelUI {
  modelId: string
  selectedNumberOfModes: number
}

export interface RegressionDifferenceData {
  caseId: string
  modelId: string
  modePairId: string
  grid: {
    [key: string]: [number, number][]
  }
}
