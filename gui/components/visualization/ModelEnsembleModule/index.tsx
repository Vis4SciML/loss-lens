"use client"

import { useEffect } from "react"
import { useAtom, useAtomValue } from "jotai"

import { fetchModelEnsembleDataAtom, showModelEnsembleAtom } from "@/lib/store"

import ModelEnsembleCore from "./ModelEnsembleCore"

export default function ModelEnsembleModule() {
  const showModelEnsemble = useAtomValue(showModelEnsembleAtom)
  const [data, fetchData] = useAtom(fetchModelEnsembleDataAtom)

  useEffect(() => {
    console.log("useEffect")
    console.log("showModelEnsemble", showModelEnsemble)
    if (showModelEnsemble) {
      console.log("fetching data in userEffect")
      fetchData()
    }
  }, [showModelEnsemble, fetchData])

  return (
    <div>
      <ModelEnsembleCore data={data} />
    </div>
  )
}
