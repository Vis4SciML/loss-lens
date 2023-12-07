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

interface LocalStructureProps {
  height: number
  width: number
}

export default function LocalStructure({ height, width }: LocalStructureProps) {
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
        <div className="col-span-5 h-full">
          <div className="grid h-full grid-cols-3">
            <div className="col-span-1 h-full">
              <LossLandscape
                height={canvasHeight / 4}
                width={canvasWidth / 4}
                modelId={ids[0]}
                modeId={ids[1]}
                leftRight={mId === 0 ? "left" : "right"}
              />
            </div>
            <div className="col-span-1 h-full">
              <PersistenceBarcode
                height={canvasHeight / 4}
                width={canvasWidth / 4}
                modelId={ids[0]}
                modeId={ids[1]}
                leftRight={mId === 0 ? "left" : "right"}
              />
            </div>
            <div className="col-span-1 h-full">
              <MergeTreeModule
                height={canvasHeight / 2}
                width={canvasWidth / 4}
                modelId={ids[0]}
                modeId={ids[1]}
                leftRight={mId === 0 ? "left" : "right"}
              />
            </div>
          </div>
        </div>
      )
    })

    return (
      <div className="grid h-[calc(22.5vh)] grid-cols-11">
        <div className="col-span-1 flex h-full items-center justify-center  font-serif text-lg">
          Local Structure
        </div>
        {modeCards}
      </div>
    )
  }
  return (
    <div className="grid h-[calc(22.5vh)] grid-cols-11">
      <div className="col-span-1 flex h-full items-center justify-center  font-serif text-lg">
        Local Structure
      </div>
      <div className="col-span-5 h-full p-1">
        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
          please select a case study to start
        </div>
      </div>
      <div className="col-span-5 h-full p-1">
        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
          please select a case study to start
        </div>
      </div>
    </div>
  )
}
