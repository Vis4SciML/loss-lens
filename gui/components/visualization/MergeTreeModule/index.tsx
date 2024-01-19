"use client"

import { useEffect } from "react"
import { useAtom } from "jotai"

import { isMergeTreeDataType } from "@/types/losslens"
import { loadMergeTreeDataAtom, systemConfigAtom } from "@/lib/losslensStore"
import { fetchCheckpointMergeTreeDataAtomFamily } from "@/lib/store"

import MergeTreeCore from "./MergeTreeCore"

interface MergeTreeProps {
  height: number
  width: number
  checkpointId: string
}

export default function MergeTree({
  height,
  width,
  checkpointId,
}: MergeTreeProps) {
  const [mergeTreeDataLoader] = useAtom(
    fetchCheckpointMergeTreeDataAtomFamily(checkpointId)
  )

  if (mergeTreeDataLoader.state === "hasError") {
    return <div>error</div>
  } else if (mergeTreeDataLoader.state === "loading") {
    return <div>loading</div>
  } else {
    if (mergeTreeDataLoader.data === null) {
      return (
        <div className={" h-[900px] w-full bg-gray-100 text-center "}>
          Merge Tree Empty
        </div>
      )
    } else {
      return (
        <MergeTreeCore
          height={height}
          width={width}
          data={mergeTreeDataLoader.data}
        />
      )
    }
  }
}
