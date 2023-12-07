"use client"

import { useEffect, useRef, useState } from "react"
import { useAtom, useAtomValue } from "jotai"

import {
  fetchSubgraphAtom,
  selectedGraphAtom,
  showNetworkAtom,
} from "@/lib/store"

import SupplyChainCore from "./SupplyChainCore"

export default function SupplyChainModule({ height, width }) {
  const showNetwork = useAtomValue(showNetworkAtom)
  const [data, fetchData] = useAtom(fetchSubgraphAtom)
  const [selecteSubGraph] = useAtom(selectedGraphAtom)

  const componentRef = useRef(null)
  const [componentWidth, setComponentWidth] = useState(0)
  const [componentHeight, setComponentHeight] = useState(0)

  // useEffect(() => {
  //   const updateComponentSize = () => {
  //     if (componentRef.current) {
  //       const { width, height } = componentRef.current.getBoundingClientRect()
  //       setComponentWidth(width)
  //       setComponentHeight(height)
  //     }
  //   }
  //
  //   updateComponentSize()
  //
  //   window.addEventListener("resize", updateComponentSize)
  //
  //   return () => {
  //     window.removeEventListener("resize", updateComponentSize)
  //   }
  // }, [])

  useEffect(() => {
    if (showNetwork) {
      fetchData()
    }
  }, [showNetwork, fetchData, selecteSubGraph])

  if (data) {
    return (
      <div className="border border-gray-100 ">
        <SupplyChainCore height={height} width={width} data={data} />
      </div>
    )
  }

  return <div>Empty</div>
}
