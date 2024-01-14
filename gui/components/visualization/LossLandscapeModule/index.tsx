"use client"

import { Suspense, useEffect } from "react"
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
      console.log("LossLandscape data not obtained")
      console.log(lossLandscapeDataLoader.data)
      console.log(globalInfoLoader.data)
      return <div className={" h-[900px] w-full text-center "}>Empty</div>
    } else {
      console.log("LossLandscape data obtained")
      console.log(lossLandscapeDataLoader.data)
      console.log(globalInfoLoader.data)
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
