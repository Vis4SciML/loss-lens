import { useEffect } from "react"
import { useAtom } from "jotai"

import { selectedCheckPointIdListAtom } from "@/lib/store"

import LossLandscape from "./visualization/LossLandscapeModule"
import MergeTreeModule from "./visualization/MergeTreeModule"
import PersistenceBarcode from "./visualization/PersistenceBarcodeModule"

interface LocalStructureProps {
  height: number
  width: number
}

export default function LocalStructure({ height, width }: LocalStructureProps) {
  const [selectedCheckPointIdList] = useAtom(selectedCheckPointIdListAtom)

  const canvasHeight = height - 170
  const canvasWidth = width - 100

  if (selectedCheckPointIdList.length !== 0) {
    const modeCards = selectedCheckPointIdList.map((checkPointId, mId) => {
      if (checkPointId === "") {
        return (
          <div className="col-span-5 h-full p-1" key={mId}>
            <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
              please select a case study to start
            </div>
          </div>
        )
      }
      return (
        <div className="col-span-5 h-full">
          <div className="grid h-full grid-cols-3">
            <div className="col-span-1 h-full">
              <LossLandscape
                height={canvasHeight / 4}
                width={canvasWidth / 4}
                checkpointId={checkPointId}
              />
            </div>
            <div className="col-span-1 h-full">
              <PersistenceBarcode
                height={canvasHeight / 4}
                width={canvasWidth / 4}
                checkpointId={checkPointId}
              />
            </div>
            <div className="col-span-1 h-full">
              <MergeTreeModule
                height={canvasHeight / 2}
                width={canvasWidth / 4}
                checkpointId={checkPointId}
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
