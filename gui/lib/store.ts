import { atom } from "jotai"
import { atomFamily, loadable, selectAtom } from "jotai/utils"

import {
  GlobalInfo,
  LossLandscape,
  SemiGlobalLocalStructure,
  SystemConfig,
} from "@/types/losslens"

import {
  fetchLossLandscapeData,
  fetchMergeTreeData,
  fetchModelMetaDataList,
  fetchPersistenceBarcodeData,
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

type LoadableState = {
  state: "loading" | "hasData" | "hasError"
  data?: any
  error?: any
}

interface UpdateCheckpointId {
  index: number
  value: string
}

export const selectedCheckPointIdListSourceAtom = atom<string[]>([])

export const selectedCheckPointIdListAtom = atom(
  (get) => {
    return get(selectedCheckPointIdListSourceAtom)
  },
  (_get, set, update: UpdateCheckpointId) => {
    set(selectedCheckPointIdListSourceAtom, (prev) => {
      let newList: string[] = []
      if (prev.length === 0) {
        newList = new Array(update.index + 1).fill("")
        newList[update.index] = update.value
      } else if (prev.length <= update.index) {
        newList = new Array(update.index + 1).fill("")
        newList[update.index] = update.value
        for (let i = 0; i < prev.length; i++) {
          newList[i] = prev[i]
        }
      } else {
        newList = [...prev]
        newList[update.index] = update.value
      }
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

export const fetchCheckpointLossLandscapeDataAtomFamily = atomFamily(
  (checkpointID: string) =>
    loadable(
      atom(async (get) => {
        const selectedCaseStudy = get(selectedCaseStudyAtom)
        console.log("fetchCheckpointLossLandscapeDataAtomFamily", checkpointID)
        if (!checkpointID) {
          return null
          // Or some default value
        }
        try {
          const data = await fetchLossLandscapeData(
            selectedCaseStudy,
            checkpointID
          ) // Your data fetching logic
          console.log("fetchCheckpointLossLandscapeDataAtomFamily", data)
          return data
        } catch (error) {
          throw error
        }
      })
    )
)

export const fetchCheckpointPersistenceBarcodeDataAtomFamily = atomFamily(
  (checkpointID: string) =>
    loadable(
      atom(async (get) => {
        const selectedCaseStudy = get(selectedCaseStudyAtom)
        if (!checkpointID) {
          return null // Or some default value
        }
        try {
          const data = await fetchPersistenceBarcodeData(
            selectedCaseStudy,
            checkpointID
          ) // Your data fetching logic
          return data
        } catch (error) {
          throw error
        }
      })
    )
)

export const fetchCheckpointMergeTreeDataAtomFamily = atomFamily(
  (checkpointID: string) =>
    loadable(
      atom(async (get) => {
        const selectedCaseStudy = get(selectedCaseStudyAtom)
        if (!checkpointID) {
          return null // Or some default value
        }
        try {
          const data = await fetchMergeTreeData(selectedCaseStudy, checkpointID) // Your data fetching logic
          return data
        } catch (error) {
          throw error
        }
      })
    )
)
