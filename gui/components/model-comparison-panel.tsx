import { useEffect } from "react"
import { useAtom } from "jotai"

import {
  loadModelMetaDataListAtom,
  selectedModeIdListAtom,
  systemConfigAtom,
} from "@/lib/losslensStore"

import ConfusionMatrixBarModule from "./visualization/ConfusionMatrixBarModule"
import LayerSimilarityModule from "./visualization/LayerSimilarityModule"
import LossLandscape from "./visualization/LossLandscapeModule"
import MergeTreeModule from "./visualization/MergeTreeModule"
import PersistenceBarcode from "./visualization/PersistenceBarcodeModule"
import RegressionDifferenceModule from "./visualization/RegressionDifferenceModule"

export default function ModelComparisonPanel({ height, width }) {
  const [systemConfig] = useAtom(systemConfigAtom)
  const [selectedModesList] = useAtom(selectedModeIdListAtom)
  const [modelMetaDataList, fetchData] = useAtom(loadModelMetaDataListAtom)

  useEffect(() => {
    if (systemConfig) {
      fetchData()
    }
  }, [systemConfig, fetchData])

  const canvasHeight = height - 170
  const canvasWidth = width - 100
  let selectedModesListStr = ""
  if (selectedModesList && selectedModesList.length === 1) {
    selectedModesListStr = selectedModesList[0]
  } else if (selectedModesList && selectedModesList.length === 2) {
    selectedModesListStr = selectedModesList[0] + " vs " + selectedModesList[1]
  }

  if (modelMetaDataList) {
    const modeCards = selectedModesList.map((modelIdModeId, mId) => {
      const ids = modelIdModeId.split("-")
      return (
        <div className="w-full">
          <div className="px-4 font-serif text-lg font-medium">
            {"Local Structure [" + modelIdModeId + "]"}
          </div>
          <div className="mt-2 w-full">
            <div className="h-4 w-full border-l border-r border-t border-black"></div>
          </div>
          <div className="flex flex-row gap-2">
            <div>
              <LossLandscape
                height={canvasHeight / 4}
                width={canvasWidth / 4}
                modelId={ids[0]}
                modeId={ids[1]}
                leftRight={mId === 0 ? "left" : "right"}
              />
              <PersistenceBarcode
                height={canvasHeight / 4}
                width={canvasWidth / 4}
                modelId={ids[0]}
                modeId={ids[1]}
                leftRight={mId === 0 ? "left" : "right"}
              />
            </div>
            <MergeTreeModule
              height={canvasHeight / 2}
              width={canvasWidth / 4}
              modelId={ids[0]}
              modeId={ids[1]}
              leftRight={mId === 0 ? "left" : "right"}
            />
          </div>
        </div>
      )
    })

    let predictionComparison = null
    if (
      selectedModesList.length === 2 &&
      systemConfig?.selectedCaseStudy === "pinn"
    ) {
      predictionComparison = (
        <div className="h-[calc(26.5vh)] ">
          <RegressionDifferenceModule
            height={canvasHeight / 2}
            width={canvasWidth / 2}
            modelIdModeIds={selectedModesList}
          />
        </div>
      )
    } else if (selectedModesList.length === 2) {
      predictionComparison = (
        <div className="h-[calc(26.5vh)] w-full ">
          <ConfusionMatrixBarModule
            height={canvasHeight / 2}
            width={canvasWidth / 2}
            modelIdModeIds={selectedModesList}
          />
        </div>
      )
    }

    const similarityPerdictions =
      selectedModesList.length === 2 ? (
        <div className="aspect-square">
          <LayerSimilarityModule
            height={canvasHeight / 2}
            width={canvasWidth / 2}
            modelIdModeIds={selectedModesList}
          />
        </div>
      ) : null

    return (
      <div className="col-span-4 h-[calc(100vh-4rem)]">
        <div className="flex h-[calc(5vh)] items-center justify-center font-serif text-lg ">
          Mode Comparison
        </div>
        {predictionComparison}
        {similarityPerdictions}
      </div>
    )
  }
  return (
    <div className="col-span-4 h-[calc(100vh-4rem)]">
      <div className="flex h-[calc(5vh)] items-center justify-center font-serif text-lg ">
        Mode Comparison
      </div>
      <div className="h-[calc(26.5vh)]"></div>
      <div className="aspect-square"></div>
    </div>
  )
}
