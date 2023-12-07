"use client"

import { useEffect } from "react"
import { useAtom, useAtomValue } from "jotai"

import {
  fetchSupplyPathAtom,
  selectedGraphAtom,
  showSupplyChainGraphAtom,
} from "@/lib/store"

import SupplyPathCore from "./SupplyPathCore"

export default function SupplyPathModule({ height, width }) {
  const showSupplyChainGraph = useAtomValue(showSupplyChainGraphAtom)
  const [data, fetchData] = useAtom(fetchSupplyPathAtom)

  const [selecteSubGraph] = useAtom(selectedGraphAtom)

  useEffect(() => {
    if (showSupplyChainGraph) {
      fetchData()
    }
  }, [showSupplyChainGraph, fetchData, selecteSubGraph])

  if (data) {
    return (
      <div className="border border-gray-100 ">
        <SupplyPathCore height={height} width={width} data={data} />
      </div>
    )
  }

  return <div>Empty</div>
}
