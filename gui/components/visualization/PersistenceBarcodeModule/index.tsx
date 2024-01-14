"use client"

import { useEffect } from "react"
import { useAtom } from "jotai"

import { isPersistenceBarcodeType } from "@/types/losslens"
import {
  loadPersistenceBarcodeDataAtom,
  systemConfigAtom,
} from "@/lib/losslensStore"
import { fetchCheckpointPersistenceBarcodeDataAtomFamily } from "@/lib/store"

import PersistenceBarcodeCore from "./PersistenceBarcodeCore"

interface PersistenceBarcodeProps {
  height: number
  width: number
  checkpointId: string
}

export default function PersistenceBarcode({
  height,
  width,
  checkpointId,
}: PersistenceBarcodeProps) {
  const [persistenceBarcodeDataLoader] = useAtom(
    fetchCheckpointPersistenceBarcodeDataAtomFamily(checkpointId)
  )

  if (persistenceBarcodeDataLoader.state === "hasError") {
    return <div>error</div>
  } else if (persistenceBarcodeDataLoader.state === "loading") {
    return <div>loading</div>
  } else {
    if (persistenceBarcodeDataLoader.data === null) {
      return (
        <div className={" h-[900px] w-full bg-gray-100 text-center "}>
          Empty
        </div>
      )
    } else {
      return (
        <PersistenceBarcodeCore
          height={height}
          width={width}
          data={persistenceBarcodeDataLoader.data}
        />
      )
    }
  }
}
