"use client"

import { useEffect } from "react"
import { useAtom } from "jotai"

import {
  loadPersistenceBarcodeData2Atom,
  loadPersistenceBarcodeDataAtom,
  systemConfigAtom,
} from "@/lib/losslensStore"

import PersistenceBarcodeCore from "./PersistenceBarcodeCore"

export default function PersistenceBarcode({
  height,
  width,
  modelId,
  modeId,
  leftRight,
}) {
  const [systemConfig] = useAtom(systemConfigAtom)
  let loadPersistenceBarcodeDataAbsAtom = loadPersistenceBarcodeDataAtom
  if (leftRight === "right") {
    loadPersistenceBarcodeDataAbsAtom = loadPersistenceBarcodeData2Atom
  }
  const [data, fetchData] = useAtom(loadPersistenceBarcodeDataAbsAtom)

  useEffect(() => {
    if (systemConfig) {
      fetchData(modelId, modeId)
    }
  }, [systemConfig, fetchData, modelId, modeId])

  if (data) {
    return <PersistenceBarcodeCore height={height} width={width} data={data} />
  }

  return <div className={" h-[900px] w-full text-center "}>Empty</div>
}
