"use client"

import { useEffect } from "react"
import { useAtom } from "jotai"

import {
  loadSemiGlobalLocalStructureAtom,
  modelUIListAtom,
  systemConfigAtom,
  updateSelectedModeIdListAtom,
} from "@/lib/losslensStore"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

import SemiGlobalLocalCore from "./SemiGlobalLocalCore"

export default function SemiGlobalLocalModule({ height, width }) {
  const [systemConfig] = useAtom(systemConfigAtom)
  const [data, fetchData] = useAtom(loadSemiGlobalLocalStructureAtom)
  const [, updateSelectedModelIdModeId] = useAtom(updateSelectedModeIdListAtom)
  const [modelUIList] = useAtom(modelUIListAtom)

  useEffect(() => {
    if (systemConfig) {
      fetchData()
    }
  }, [systemConfig, fetchData])
  const canvasHeight = height - 170
  const canvasWidth = width - 30

  if (data && modelUIList?.length > 0) {
    return (
      <div className="grid grid-cols-11 ">
        <div className="col-span-1 flex h-full items-center justify-center  font-serif text-lg">
          Global Structure
        </div>
        <div className="col-span-5 aspect-square p-1">
          <SemiGlobalLocalCore
            height={canvasHeight / 2}
            width={canvasWidth}
            data={data}
            updateSelectedModelIdModeId={updateSelectedModelIdModeId}
            modelUIList={modelUIList}
            modelUI={modelUIList[0]}
          />
        </div>
        <div className="col-span-5 aspect-square p-1">
          <SemiGlobalLocalCore
            height={canvasHeight / 2}
            width={canvasWidth}
            data={data}
            updateSelectedModelIdModeId={updateSelectedModelIdModeId}
            modelUIList={modelUIList}
            modelUI={modelUIList[1]}
          />
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-11 ">
      <div className="col-span-1 flex h-full items-center justify-center  font-serif text-lg">
        Global Structure
      </div>
      <div className="col-span-5  aspect-square  p-1 ">
        <div className=" flex h-full w-full items-center justify-center rounded border text-gray-500">
          please select a case study to start
        </div>
      </div>
      <div className="col-span-5  aspect-square  p-1">
        <div className=" flex h-full w-full items-center justify-center rounded border text-gray-500">
          please select a case study to start
        </div>
      </div>
    </div>
  )
}
