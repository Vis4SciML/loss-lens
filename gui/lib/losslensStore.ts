import { DropdownMenuCheckboxItemProps } from "@radix-ui/react-dropdown-menu"
import { atom } from "jotai"
import { selectAtom } from "jotai/utils"

import {
  ConfusionMaterixBarData,
  GlobalInfo,
  isModelMetaData,
  LayerSimilarityData,
  LossLandscape,
  MergeTreeData,
  ModelMetaData,
  ModelUI,
  PersistenceBarcode,
  RegressionDifferenceData,
  SemiGlobalLocalStructure,
  SystemConfig,
} from "@/types/losslens"
import {
  fetchLayerSimilarityData,
  fetchLossLandscapeData,
  fetchMergeTreeData,
  fetchModelMetaDataList,
  fetchPersistenceBarcodeData,
  fetchRegressionDifferenceData,
  fetchSemiGlobalLocalStructureData,
} from "@/lib/api"

import { fetchConfusionMatrixBarData } from "./api"

/**
 * System Config & UI
 */

export const systemConfigAtom = atom<SystemConfig>({
  caseStudyList: ["vit", "resnet20", "pinn"],
  caseStudyLabels: {
    // mlp: "MLP-MNSIT",
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

/**
 *  Semi-Global Local Structure
 */

export const semiGlobalLocalStructureAtom = atom<
  SemiGlobalLocalStructure | Promise<SemiGlobalLocalStructure> | null
>(null)

export const loadSemiGlobalLocalStructureAtom = atom(
  async (get) => get(semiGlobalLocalStructureAtom),
  async (get, set) => {
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    if (!selectedCaseStudy) return
    const promise = fetchSemiGlobalLocalStructureData(selectedCaseStudy).then(
      (data: SemiGlobalLocalStructure) => {
        return data
      }
    )
    set(semiGlobalLocalStructureAtom, promise)
  }
)

/**
 *  Model Meta Data List
 */

export const globalInfoAtom = atom<GlobalInfo | Promise<GlobalInfo> | null>(
  null
)

export const getGlobalInfoAtom = atom(async (get) => get(globalInfoAtom))

export const modelMetaDataListAtom = atom<
  ModelMetaData[] | Promise<ModelMetaData[]> | null
>(null)

export const loadModelMetaDataListAtom = atom(
  async (get) => get(modelMetaDataListAtom),
  async (get, set) => {
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    if (!selectedCaseStudy) return
    const promise = fetchModelMetaDataList(selectedCaseStudy).then(
      (data: { data: ModelMetaData[] }) => {
        return data.data
      }
    )
    set(modelMetaDataListAtom, promise)
  }
)

export const loadGlobalInfoAtom = atom(
  async (get) => get(globalInfoAtom),
  async (get, set) => {
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    if (!selectedCaseStudy) return
    const promise = fetchModelMetaDataList(selectedCaseStudy).then(
      (data: GlobalInfo) => {
        return data
      }
    )
    set(globalInfoAtom, promise)
  }
)

/**
 * Model UI List
 */

export const modelUIListAtom = atom<
  ModelUI[] | Promise<ModelUI[]> | undefined | null
>(null)

export const loadModelUIListAtom = atom(
  async (get) => get(modelUIListAtom),
  async (get, set) => {
    const modelMetaDataList = await get(modelMetaDataListAtom)
    const modelUIList = modelMetaDataList?.map((modelMetaData) => {
      return {
        modelId: modelMetaData.modelId,
        selectedNumberOfModes: modelMetaData.numberOfModes,
      }
    })
    set(modelUIListAtom, modelUIList)
  }
)

export const updateModelUIListAtom = atom(
  null,
  (get, set, selectedNumberOfModes: number, index: number) => {
    const modelUIList = get(modelUIListAtom)
    let newModelUIList = modelUIList
    if (modelUIList) {
      newModelUIList = [...modelUIList]
      newModelUIList[index].selectedNumberOfModes = selectedNumberOfModes
      set(modelUIListAtom, newModelUIList)
    }
  }
)

/**
 * Selected Mode Id List
 */

export const selectedModeIdListAtom = atom<string[]>([])
export const updateSelectedModeIdListAtom = atom(
  null,
  async (get, set, modelIdModeId: string) => {
    let selectedModeIdList = get(selectedModeIdListAtom)
    const modelMetaDataList = get(modelMetaDataListAtom)

    if (!modelMetaDataList) return

    // initialize selectedModeIdList
    if (selectedModeIdList.length !== modelMetaDataList.value.length) {
      selectedModeIdList = modelMetaDataList.value.map((_modelMetaData) => "")
    }

    modelMetaDataList.value.forEach((modelMetaData, index) => {
      if (modelMetaData.modelId === modelIdModeId.split("-")[0]) {
        if (selectedModeIdList[index] === modelIdModeId) {
          selectedModeIdList[index] = ""
        } else {
          selectedModeIdList[index] = modelIdModeId
        }
      }
    })
    set(selectedModeIdListAtom, selectedModeIdList)
  }
)

/**
 * Loss Landscape Data
 */

type LossLandscapeDataAtomType = {
  [key: string]: LossLandscape | Promise<LossLandscape> | null
}

const createLossLandscapeDataAtom = () => {
  const baseAtom = atom<LossLandscapeDataAtomType>({})
  const valueAtom = atom(async (get) => get(baseAtom))
  const loadAtom = atom(null, async (get, set, modelIdModeId: string) => {
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    const selectedModeIdList = get(selectedModeIdListAtom)
    const lossLandscapeData = get(baseAtom)
    if (!selectedCaseStudy || !selectedModeIdList || modelIdModeId === "")
      return
    const promise = fetchLossLandscapeData(
      selectedCaseStudy,
      modelIdModeId
    ).then((data: LossLandscape) => {
      return data
    })
    lossLandscapeData[modelIdModeId] = promise
    const newData = { ...lossLandscapeData }

    set(baseAtom, newData)
  })
  return [valueAtom, loadAtom]
}

export const [lossLandscapeDataAtom, loadLossLandscapeDataAtom] =
  createLossLandscapeDataAtom()

// // to delete
// export const [lossLandscapeDataAtom1, loadLossLandscapeDataAtom1] =
//   createLossLandscapeDataAtom()
//
// export const [lossLandscapeDataAtom2, loadLossLandscapeDataAtom2] =
//   createLossLandscapeDataAtom()
//
// export const twoLossLandscapeLossScaleAtom = atom(async (get) => {
//   const lossLandscapeData1 = get(lossLandscapeDataAtom1)
//   const lossLandscapeData2 = get(lossLandscapeDataAtom2)
//   if (!lossLandscapeData1 || !lossLandscapeData2) return
//   console.log(lossLandscapeData1, lossLandscapeData2)
//   return 214
// })

/**
 * Persistence Barcode Data
 */

type PersistenceBarcodeDataAtomType = {
  [key: string]: PersistenceBarcode | Promise<PersistenceBarcode> | null
}

export const persistenceBarcodeDataAtom = atom<PersistenceBarcodeDataAtomType>(
  {}
)

export const loadPersistenceBarcodeDataAtom = atom(
  async (get) => get(persistenceBarcodeDataAtom),
  async (get, set, modelIdModeId: string) => {
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    const persistenceBarcodeData = get(persistenceBarcodeDataAtom)
    if (!selectedCaseStudy || modelIdModeId === "") return
    const promise = fetchPersistenceBarcodeData(
      selectedCaseStudy,
      modelIdModeId
    ).then((data: PersistenceBarcode) => {
      return data
    })
    persistenceBarcodeData[modelIdModeId] = promise
    set(persistenceBarcodeDataAtom, persistenceBarcodeData)
  }
)

// to delete
// export const persistenceBarcodeData2Atom = atom<
//   PersistenceBarcode | Promise<PersistenceBarcode> | null
// >(null)
//
// export const loadPersistenceBarcodeData2Atom = atom(
//   async (get) => get(persistenceBarcodeData2Atom),
//   async (get, set, modelIdModeId: string) => {
//     const selectedCaseStudy = get(selectedCaseStudyAtom)
//     if (!selectedCaseStudy) return
//     const promise = fetchPersistenceBarcodeData(
//       selectedCaseStudy,
//       modelIdModeId
//     ).then((data: PersistenceBarcode) => {
//       return data
//     })
//     set(persistenceBarcodeData2Atom, promise)
//   }
// )

/**
 * Merge Tree Data
 */

type MergeTreeDataAtomType = {
  [key: string]: MergeTreeData | Promise<MergeTreeData> | null
}
export const mergeTreeDataAtom = atom<MergeTreeDataAtomType>({})

export const loadMergeTreeDataAtom = atom(
  async (get) => get(mergeTreeDataAtom),
  async (get, set, modelIdModeId: string) => {
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    const mergeTreeData = get(mergeTreeDataAtom)
    if (!selectedCaseStudy || modelIdModeId === "") return
    const promise = fetchMergeTreeData(selectedCaseStudy, modelIdModeId).then(
      (data: MergeTreeData) => {
        return data
      }
    )
    mergeTreeData[modelIdModeId] = promise
    set(mergeTreeDataAtom, mergeTreeData)
  }
)

// to delete
// export const mergeTreeData2Atom = atom<
//   MergeTreeData | Promise<MergeTreeData> | null
// >(null)
//
// export const loadMergeTreeData2Atom = atom(
//   async (get) => get(mergeTreeData2Atom),
//   async (get, set, modelIdModeId: string) => {
//     const selectedCaseStudy = get(selectedCaseStudyAtom)
//     if (!selectedCaseStudy) return
//     const promise = fetchMergeTreeData(selectedCaseStudy, modelIdModeId).then(
//       (data: MergeTreeData) => {
//         return data
//       }
//     )
//     set(mergeTreeData2Atom, promise)
//   }
// )

/**
 * Layer Similarity data
 */

export const layerSimilarityDataAtom = atom<
  LayerSimilarityData | Promise<LayerSimilarityData> | null
>(null)

export const loadLayerSimilarityDataAtom = atom(
  async (get) => get(layerSimilarityDataAtom),
  async (get, set) => {
    const selectedModeIdList = get(selectedModeIdListAtom)
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    if (!selectedCaseStudy || selectedModeIdList.includes("")) return
    const promise = fetchLayerSimilarityData(
      selectedCaseStudy,
      selectedModeIdList
    ).then((data: LayerSimilarityData) => {
      const selectedXLabels = data.xLabels.map((xLabel) => {
        if (data.caseId === "resnet20" && xLabel.includes("conv")) {
          return true
        } else if (data.caseId === "vit" && xLabel.includes("transformer")) {
          return true
        } else if (data.caseId === "pinn" && xLabel !== "") {
          return true
        }
        return false
      })
      const selectedYLabels = data.yLabels.map((yLabel) => {
        if (data.caseId === "resnet20" && yLabel.includes("conv")) {
          return true
        } else if (data.caseId === "vit" && yLabel.includes("transformer")) {
          return true
        } else if (data.caseId === "pinn" && yLabel !== "") {
          return true
        }

        return false
      })
      set(XCheckBoxAtom, selectedXLabels)
      set(YCheckBoxAtom, selectedYLabels)
      return data
    })
    set(layerSimilarityDataAtom, promise)
  }
)

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

export const confusionMatrixBarDataAtom = atom<
  ConfusionMaterixBarData | Promise<ConfusionMaterixBarData> | null
>(null)

export const loadConfusionMatrixBarDataAtom = atom(
  async (get) => get(confusionMatrixBarDataAtom),
  async (get, set, modelIdModeIds: string[]) => {
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    if (!selectedCaseStudy) return
    const promise = fetchConfusionMatrixBarData(
      selectedCaseStudy,
      modelIdModeIds
    ).then((data: ConfusionMaterixBarData) => {
      return data
    })
    set(confusionMatrixBarDataAtom, promise)
  }
)

/**
 * Regression Difference Data
 */

export const regressionDifferenceDataAtom = atom<
  RegressionDifferenceData | Promise<RegressionDifferenceData> | null
>(null)

export const loadRegressionDifferenceDataAtom = atom(
  async (get) => get(regressionDifferenceDataAtom),
  async (get, set, modelIdModeIds: string[]) => {
    const selectedCaseStudy = get(selectedCaseStudyAtom)
    if (!selectedCaseStudy) return
    const promise = fetchRegressionDifferenceData(
      selectedCaseStudy,
      modelIdModeIds
    ).then((data: RegressionDifferenceData) => {
      return data
    })
    set(regressionDifferenceDataAtom, promise)
  }
)
