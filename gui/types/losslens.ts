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

export function isModelMetaData(obj: any): obj is ModelMetaData {
  return (
    obj &&
    obj.modelId &&
    obj.modelName &&
    obj.modelDescription &&
    obj.modelDataset &&
    obj.datasetId &&
    obj.modelDatasetDescription &&
    obj.numberOfModes
  )
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

export function isLossLandscapeType(obj: any): obj is LossLandscape {
  return (
    obj &&
    obj.caseId &&
    obj.modelId &&
    obj.modeId &&
    obj.grid &&
    obj.upperBound &&
    obj.lowerBound
  )
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

export function isPersistenceBarcodeType(obj: any): obj is PersistenceBarcode {
  return obj && obj.caseId && obj.modelId && obj.modeId && obj.edges
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

export function isMergeTreeDataType(obj: any): obj is MergeTreeData {
  return obj && obj.modeId && obj.nodes && obj.edges
}

export interface LayerSimilarityData {
  caseId: string
  modePairId: string
  modelX: string
  modelY: string
  checkPointX: string
  checkPointY: string
  grid: {
    xId: number
    yId: number
    value: number
  }[]
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
