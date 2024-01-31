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
        <div className="grid grid-cols-4">
          <div className="col-span-3">
            <div className="h-[calc(26.5vh)]"></div>
            <div className="aspect-square"></div>
          </div>
          <div className="col-span-1 flex h-[calc(100vh-5rem)] items-center justify-center font-serif text-lg ">
            <div>Model Comparison</div>
          </div>
        </div>
      </div>
    )
  } else {
    let predictionComparison = null
    if (selectedCaseStudy === "pinn") {
      predictionComparison = (
        <div className="aspect-square ">
          <RegressionDifferenceModule
            height={canvasHeight / 2}
            width={canvasWidth / 2}
          />
        </div>
      )
    } else {
      predictionComparison = (
        <div className="aspect-square ">
          <ConfusionMatrixBarModule
            height={canvasHeight / 2}
            width={canvasWidth / 2}
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
        <div className="grid grid-cols-4 p-1">
          <div className="col-span-3 flex flex-col justify-around rounded border">
            {predictionComparison}
            {similarityPerdictions}
          </div>
          <div className="col-span-1 flex h-[calc(100vh-5rem)] flex-col  justify-around font-serif text-lg ">
            <div className="p-2">Prediction Disparity View</div>
            <div className="p-2">Layer Similarity View</div>
          </div>
        </div>
      </div>
    )
  }
}
