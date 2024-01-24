import { DropdownMenuCheckboxItemProps } from "@radix-ui/react-dropdown-menu"
import { atom } from "jotai"
import { atomFamily, loadable, selectAtom } from "jotai/utils"

import {
  ConfusionMaterixBarData,
  LayerSimilarityData,
  RegressionDifferenceData,
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

type LoadableState = {
  state: "loading" | "hasData" | "hasError"
  data?: any
  error?: any
}

interface UpdateCheckpointId {
  index: number
  value: string
}

export const selectedCheckPointIdListSourceAtom = atom<string[]>(["", ""])

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

/**
 * Layer Similarity data
 */

export const layerSimilarityDataAtom = atom(async (get) => {
  const selectedCheckPointIdList = get(selectedCheckPointIdListSourceAtom)
  if (
    selectedCheckPointIdList[0] === "" ||
    selectedCheckPointIdList[1] === ""
  ) {
    return null
  } else {
    try {
      const data = await fetchLayerSimilarityData(
        get(selectedCaseStudyAtom),
        selectedCheckPointIdList
      )
      return data
    } catch (error) {
      throw error
    }
  }
})

export const loadableLayerSimilarityDataAtom = loadable(layerSimilarityDataAtom)

// export const loadLayerSimilarityDataAtom = atom(
//   async (get) => get(layerSimilarityDataAtom),
//   async (get, set) => {
//     const selectedModeIdList = get(selectedModeIdListAtom)
//     const selectedCaseStudy = get(selectedCaseStudyAtom)
//     if (!selectedCaseStudy || selectedModeIdList.includes("")) return
//     const promise = fetchLayerSimilarityData(
//       selectedCaseStudy,
//       selectedModeIdList
//     ).then((data: LayerSimilarityData) => {
//       const selectedXLabels = data.xLabels.map((xLabel) => {
//         if (data.caseId === "resnet20" && xLabel.includes("conv")) {
//           return true
//         } else if (data.caseId === "vit" && xLabel.includes("transformer")) {
//           return true
//         } else if (data.caseId === "pinn" && xLabel !== "") {
//           return true
//         }
//         return false
//       })
//       const selectedYLabels = data.yLabels.map((yLabel) => {
//         if (data.caseId === "resnet20" && yLabel.includes("conv")) {
//           return true
//         } else if (data.caseId === "vit" && yLabel.includes("transformer")) {
//           return true
//         } else if (data.caseId === "pinn" && yLabel !== "") {
//           return true
//         }
//
//         return false
//       })
//       set(XCheckBoxAtom, selectedXLabels)
//       set(YCheckBoxAtom, selectedYLabels)
//       return data
//     })
//     set(layerSimilarityDataAtom, promise)
//   }
// )
//
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

/**
 * Confusion Matrix Bar data
 */

export const confusionMatrixBarDataAtom = atom(async (get) => {
  const selectedCheckPointIdList = get(selectedCheckPointIdListSourceAtom)
  if (
    selectedCheckPointIdList[0] === "" ||
    selectedCheckPointIdList[1] === ""
  ) {
    return null
  } else {
    try {
      const data = await fetchConfusionMatrixBarData(
        get(selectedCaseStudyAtom),
        selectedCheckPointIdList
      )
      return data
    } catch (error) {
      throw error
    }
  }
})

export const loadableConfusionMatrixBarDataAtom = loadable(
  confusionMatrixBarDataAtom
)

// export const loadConfusionMatrixBarDataAtom = atom(
//   async (get) => get(confusionMatrixBarDataAtom),
//   async (get, set, modelIdModeIds: string[]) => {
//     const selectedCaseStudy = get(selectedCaseStudyAtom)
//     if (!selectedCaseStudy) return
//     const promise = fetchConfusionMatrixBarData(
//       selectedCaseStudy,
//       modelIdModeIds
//     ).then((data: ConfusionMaterixBarData) => {
//       return data
//     })
//     set(confusionMatrixBarDataAtom, promise)
//   }
// )

/**
 * Regression Difference Data
 */

export const regressionDifferenceDataAtom = atom(async (get) => {
  const selectedCheckPointIdList = get(selectedCheckPointIdListSourceAtom)
  if (
    selectedCheckPointIdList[0] === "" ||
    selectedCheckPointIdList[1] === ""
  ) {
    return null
  } else {
    try {
      const data = await fetchRegressionDifferenceData(
        get(selectedCaseStudyAtom),
        selectedCheckPointIdList
      )
      return data
    } catch (error) {
      throw error
    }
  }
})

export const loadableRegressionDifferenceDataAtom = loadable(
  regressionDifferenceDataAtom
)

// export const loadRegressionDifferenceDataAtom = atom(
//   async (get) => get(regressionDifferenceDataAtom),
//   async (get, set, modelIdModeIds: string[]) => {
//     const selectedCaseStudy = get(selectedCaseStudyAtom)
//     if (!selectedCaseStudy) return
//     const promise = fetchRegressionDifferenceData(
//       selectedCaseStudy,
//       modelIdModeIds
//     ).then((data: RegressionDifferenceData) => {
//       return data
//     })
//     set(regressionDifferenceDataAtom, promise)
//   }
// )
