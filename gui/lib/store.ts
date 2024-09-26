import { DropdownMenuCheckboxItemProps } from "@radix-ui/react-dropdown-menu"
import { atom } from "jotai"
import { atomFamily, loadable, selectAtom } from "jotai/utils"

import {
  SystemConfig,
} from "@/types/losslens"

import {
  fetchConfusionMatrixBarData,
  fetchLayerSimilarityData,
  fetchLossLandscapeData,
  fetchMergeTreeData,
  fetchModelMetaDataList,
  fetchPersistenceBarcodeData,
  fetchRegressionDifferenceData,
  fetchSemiGlobalLocalStructureData,
} from "./api"

export const systemConfigAtom = atom<SystemConfig>({
  caseStudyList: ["vit", "resnet20", "pinn"],
  caseStudyLabels: {
    vit: "ViT-CIFAR10",
    pinn: "PINN",
    resnet20: "ResNet20-CIFAR10",
  },
  selectedCaseStudy: null,
})

export const selectedCaseStudyAtom = selectAtom(
  systemConfigAtom,
  (s) => s.selectedCaseStudy
)

export const applySystemConfigAtom = atom(null, async (get, set) => {
  const systemConfig = get(systemConfigAtom)
  set(systemConfigAtom, systemConfig)
})

const fetchModelIDsAtom = atom(async (get) => {
  const selectedCaseStudy = get(selectedCaseStudyAtom)
  const response = await fetchModelMetaDataList(selectedCaseStudy)
  return response
})

export const modelIDLoadableAtom = loadable(fetchModelIDsAtom)


interface UpdateCheckpointId {
  index: number
  value: string
}

export const selectedCheckPointIdListSourceAtom = atom<string[]>(["", ""])

export const selectedCheckPointIdListAtom = atom(
  (get) => get(selectedCheckPointIdListSourceAtom),
  (_get, set, update: UpdateCheckpointId) => {
    set(selectedCheckPointIdListSourceAtom, (prev) => {
      const newList = [...prev]
      newList[update.index] = update.value
      return newList
    })
  }
)

/**
 *  Semi-Global Local Structure
 */

export const loadSemiGlobalLocalStructureAtom = atom(async (get) => {
  const selectedCaseStudy = get(selectedCaseStudyAtom)
  if (!selectedCaseStudy) return null
  try {
    const data = await fetchSemiGlobalLocalStructureData(selectedCaseStudy)
    return data
  } catch (error) {
    throw error
  }
})

export const loadableSemiGlobalLocalStructureAtom = loadable(
  loadSemiGlobalLocalStructureAtom
)

const createFetchCheckpointDataAtomFamily = (fetchDataFunc) => atomFamily(
  (checkpointID: string) =>
    loadable(
      atom(async (get) => {
        const selectedCaseStudy = get(selectedCaseStudyAtom)
        if (!checkpointID) return null
        try {
          const data = await fetchDataFunc(selectedCaseStudy, checkpointID)
          return data
        } catch (error) {
          throw error
        }
      })
    )
)

export const fetchCheckpointLossLandscapeDataAtomFamily = createFetchCheckpointDataAtomFamily(fetchLossLandscapeData)
export const fetchCheckpointPersistenceBarcodeDataAtomFamily = createFetchCheckpointDataAtomFamily(fetchPersistenceBarcodeData)
export const fetchCheckpointMergeTreeDataAtomFamily = createFetchCheckpointDataAtomFamily(fetchMergeTreeData)

/**
 * Layer Similarity data
 */

const createLoadableDataAtom = (fetchDataFunc: (caseStudy: any, checkPointIdList: string[]) => Promise<any>) => atom(async (get) => {
  const selectedCheckPointIdList = get(selectedCheckPointIdListSourceAtom)
  if (selectedCheckPointIdList.includes("")) return null
  try {
    const data = await fetchDataFunc(get(selectedCaseStudyAtom), selectedCheckPointIdList)
    return data
  } catch (error) {
    throw error
  }
})

export const layerSimilarityDataAtom = createLoadableDataAtom(fetchLayerSimilarityData)
export const loadableLayerSimilarityDataAtom = loadable(layerSimilarityDataAtom)

/**
 * Confusion Matrix Bar data
 */

export const confusionMatrixBarDataAtom = createLoadableDataAtom(fetchConfusionMatrixBarData)
export const loadableConfusionMatrixBarDataAtom = loadable(confusionMatrixBarDataAtom)

/**
 * Regression Difference Data
 */

export const regressionDifferenceDataAtom = createLoadableDataAtom(fetchRegressionDifferenceData)
export const loadableRegressionDifferenceDataAtom = loadable(regressionDifferenceDataAtom)

type Checked = DropdownMenuCheckboxItemProps["checked"]

export const XCheckBoxAtom = atom<Checked[]>([])
export const YCheckBoxAtom = atom<Checked[]>([])

export const updateXCheckBoxAtom = atom(
  async (get) => get(XCheckBoxAtom),
  async (get, set) => {}
)

export const updateYCheckBoxAtom = atom(
  async (get) => get(YCheckBoxAtom),
  async (get, set) => {}
)
