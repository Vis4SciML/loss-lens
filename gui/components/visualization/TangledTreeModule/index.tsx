"use client"

import { useEffect } from "react"
import { useAtom, useAtomValue } from "jotai"

import { fetchSupplyChainDataAtom, showSupplyChainGraphAtom } from "@/lib/store"

import TangledTreeCore from "./TangledTreeCore"

export default function TangledTreeModule() {
  const showSupplyChainGraph = useAtomValue(showSupplyChainGraphAtom)
  const [data, fetchData] = useAtom(fetchSupplyChainDataAtom)

  useEffect(() => {
    if (showSupplyChainGraph) {
      fetchData()
    }
  }, [showSupplyChainGraph, fetchData])

  if (data) {
    console.log("data")
    console.log(data)
    return (
      <div className="border border-gray-100 ">
        <TangledTreeCore data={data} />
      </div>
    )
  }

  return <div>Empty</div>
}
