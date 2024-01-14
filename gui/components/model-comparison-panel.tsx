import { Play } from "next/font/google"
import { useAtom } from "jotai"

import {
  selectedCaseStudyAtom,
  selectedCheckPointIdListAtom,
} from "@/lib/store"

import ConfusionMatrixBarModule from "./visualization/ConfusionMatrixBarModule"
import LayerSimilarityModule from "./visualization/LayerSimilarityModule"
import RegressionDifferenceModule from "./visualization/RegressionDifferenceModule"

export default function ModelComparisonPanel({ height, width }) {
  const [selectedCheckpointIdList] = useAtom(selectedCheckPointIdListAtom)
  const [selectedCaseStudy] = useAtom(selectedCaseStudyAtom)

  const canvasHeight = height - 170
  const canvasWidth = width - 100
  if (
    selectedCheckpointIdList[0] === "" ||
    selectedCheckpointIdList[1] === ""
  ) {
    return (
      <div className="col-span-4 h-[calc(100vh-4rem)]">
        <div className="flex h-[calc(5vh)] items-center justify-center font-serif text-lg ">
          Mode Comparison
        </div>
        <div className="h-[calc(26.5vh)]"></div>
        <div className="aspect-square"></div>
      </div>
    )
  } else {
    let predictionComparison = null
    if (selectedCaseStudy === "pinn") {
      predictionComparison = (
        <div className="h-[calc(26.5vh)] ">
          <RegressionDifferenceModule
            height={canvasHeight / 2}
            width={canvasWidth / 2}
            modelIdModeIds={selectedCheckpointIdList}
          />
        </div>
      )
    } else {
      predictionComparison = (
        <div className="h-[calc(26.5vh)] w-full ">
          <ConfusionMatrixBarModule
            height={canvasHeight / 2}
            width={canvasWidth / 2}
            modelIdModeIds={selectedCheckpointIdList}
          />
        </div>
      )
    }

    const similarityPerdictions = (
      <div className="aspect-square">
        <LayerSimilarityModule
          height={canvasHeight / 2}
          width={canvasWidth / 2}
          modelIdModeIds={selectedCheckpointIdList}
        />
      </div>
    )

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
}
