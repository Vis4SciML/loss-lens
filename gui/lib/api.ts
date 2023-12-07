import axios from "axios"

export const fetchSemiGlobalLocalStructureData = async (id: string) => {
  const response = await axios.get(
    `api/semi-global-local-structure-data?id=${id}`
  )
  return await response.data
}

export const fetchModelMetaDataList = async (id: string) => {
  const response = await axios.get(`api/model-metadata?id=${id}`)
  return await response.data
}

export const fetchLossLandscapeData = async (
  id: string,
  modelId: string,
  modeId: string
) => {
  const response = await axios.get(
    `api/loss-landscape-data?id=${id}&modelId=${modelId}&modeId=${modeId}`
  )
  return await response.data
}

export const fetchPersistenceBarcodeData = async (
  id: string,
  modelId: string,
  modeId: string
) => {
  const response = await axios.get(
    `api/persistence-barcode-data?id=${id}&modelId=${modelId}&modeId=${modeId}`
  )
  return await response.data
}

export const fetchMergeTreeData = async (
  id: string,
  modelId: string,
  modeId: string
) => {
  const response = await axios.get(
    `api/merge-tree-data?id=${id}&modelId=${modelId}&modeId=${modeId}`
  )
  return await response.data
}

export const fetchLayerSimilarityData = async (
  id: string,
  modelIdModeIds: string[]
) => {
  const modelIdModeIdsParts = modelIdModeIds.map((modelIdModeId) =>
    modelIdModeId.split("-")
  )
  const modelId0 = modelIdModeIdsParts[0][0]
  const modeId0 = modelIdModeIdsParts[0][1]
  const modelId1 = modelIdModeIdsParts[1][0]
  const modeId1 = modelIdModeIdsParts[1][1]
  const response = await axios.get(
    `api/layer-similarity-data?id=${id}&modelId0=${modelId0}&modeId0=${modeId0}&modelId1=${modelId1}&modeId1=${modeId1}`
  )
  return await response.data
}

export const fetchConfusionMatrixBarData = async (
  id: string,
  modelIdModeIds: string[]
) => {
  const modelIdModeIdsParts = modelIdModeIds.map((modelIdModeId) =>
    modelIdModeId.split("-")
  )
  const modelId0 = modelIdModeIdsParts[0][0]
  const modeId0 = modelIdModeIdsParts[0][1]
  const modelId1 = modelIdModeIdsParts[1][0]
  const modeId1 = modelIdModeIdsParts[1][1]
  const response = await axios.get(
    `api/confusion-matrix-bar-data?id=${id}&modelId0=${modelId0}&modeId0=${modeId0}&modelId1=${modelId1}&modeId1=${modeId1}`
  )
  return await response.data
}

export const fetchRegressionDifferenceData = async (
  id: string,
  modelIdModeIds: string[]
) => {
  const modelIdModeIdsParts = modelIdModeIds.map((modelIdModeId) =>
    modelIdModeId.split("-")
  )
  const modelId0 = modelIdModeIdsParts[0][0]
  const modeId0 = modelIdModeIdsParts[0][1]
  const modelId1 = modelIdModeIdsParts[1][0]
  const modeId1 = modelIdModeIdsParts[1][1]
  const response = await axios.get(
    `api/regression-difference-data?id=${id}&modelId0=${modelId0}&modeId0=${modeId0}&modelId1=${modelId1}&modeId1=${modeId1}`
  )
  return await response.data
}
