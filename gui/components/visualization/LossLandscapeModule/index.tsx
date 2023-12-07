"use client"

import { useEffect } from "react"
import { useAtom } from "jotai"

import {
  loadGlobalInfoAtom,
  loadLossLandscapeDataAtom1,
  loadLossLandscapeDataAtom2,
  lossLandscapeDataAtom1,
  lossLandscapeDataAtom2,
  systemConfigAtom,
} from "@/lib/losslensStore"

import LossContourCore from "./LossContour"
import LossHeatMapCore from "./LossHeatMapCore"

export default function LossLandscape({
  height,
  width,
  modelId,
  modeId,
  leftRight,
}) {
  const [systemConfig] = useAtom(systemConfigAtom)
  const [globalInfo, loadGlobalInfo] = useAtom(loadGlobalInfoAtom)

  let lossLandscapeDataAbsAtom = lossLandscapeDataAtom1
  let loadLossLandscapeDataAbsAtom = loadLossLandscapeDataAtom1

  if (leftRight === "right") {
    lossLandscapeDataAbsAtom = lossLandscapeDataAtom2
    loadLossLandscapeDataAbsAtom = loadLossLandscapeDataAtom2
  }
  const [data] = useAtom(lossLandscapeDataAbsAtom)

  const [, fetchData] = useAtom(loadLossLandscapeDataAbsAtom)

  useEffect(() => {
    if (systemConfig) {
      fetchData(modelId, modeId)
      loadGlobalInfo()
    }
  }, [systemConfig, fetchData, modelId, modeId, loadGlobalInfo])

  if (data && globalInfo) {
    return (
      <LossContourCore
        height={height}
        width={width}
        data={data}
        globalInfo={globalInfo}
      />
    )
  }

  return <div className={" h-[900px] w-full text-center "}>Empty</div>
}
