"use client"

import { useAtom } from "jotai"

import {
  fetchCheckpointLossLandscapeDataAtomFamily,
  modelIDLoadableAtom,
} from "@/lib/store"

import LossContourCore from "./LossContour"

interface LossLandscapeProps {
  height: number
  width: number
  checkpointId: string
}

export default function LossLandscape({
  height,
  width,
  checkpointId,
}: LossLandscapeProps) {
  const [lossLandscapeDataLoader] = useAtom(
    fetchCheckpointLossLandscapeDataAtomFamily(checkpointId)
  )
  const [globalInfoLoader] = useAtom(modelIDLoadableAtom)

  if (
    lossLandscapeDataLoader.state === "hasError" ||
    globalInfoLoader.state === "hasError"
  ) {
    return <div>error</div>
  } else if (
    lossLandscapeDataLoader.state === "loading" ||
    globalInfoLoader.state === "loading"
  ) {
    return <div>loading</div>
  } else {
    if (
      lossLandscapeDataLoader.data === null ||
      globalInfoLoader.data === null
    ) {
      return (
        <div className={" h-[900px] w-full text-center "}>
          LossContour is empty
        </div>
      )
    } else {
      return (
        <LossContourCore
          height={height}
          width={width}
          data={lossLandscapeDataLoader.data}
          globalInfo={globalInfoLoader.data}
        />
      )
    }
  }
}
