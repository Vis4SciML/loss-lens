"use client"

import { useEffect } from "react"
import { useAtom } from "jotai"

import {
  loadConfusionMatrixBarDataAtom,
  loadRegressionDifferenceDataAtom,
  systemConfigAtom,
} from "@/lib/losslensStore"

import RegressionDifferenceCore from "./RegressionDifferenceCore"

export default function RegressionDifferenceModule({
  height,
  width,
  modelIdModeIds,
}) {
  const [systemConfig] = useAtom(systemConfigAtom)
  const [data, fetchData] = useAtom(loadRegressionDifferenceDataAtom)

  useEffect(() => {
    if (systemConfig) {
      fetchData(modelIdModeIds)
    }
  }, [systemConfig, fetchData, modelIdModeIds])

  if (data) {
    return (
      <RegressionDifferenceCore height={height} width={width} data={data} />
    )
  }

  return (
    <div
      className={"flex h-[550px] w-full flex-col justify-center text-center "}
    >
      Regression Difference View is currently not available.
    </div>
  )
}
