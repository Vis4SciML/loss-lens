"use client"

import { useAtom } from "jotai"

import { loadableConfusionMatrixBarDataAtom } from "@/lib/store"

import ConfusionMatrixBarCore from "./ConfusionMatrixBarCore"

export default function ConfusionMatrixBarModule({ height, width }) {
  const [confusionMatrixDataLoader] = useAtom(
    loadableConfusionMatrixBarDataAtom
  )

  if (confusionMatrixDataLoader.state === "hasError") {
    return <div>error</div>
  } else if (confusionMatrixDataLoader.state === "loading") {
    return <div>loading</div>
  } else {
    if (confusionMatrixDataLoader.data === null) {
      return (
        <div
          className={
            "flex h-[550px] w-full flex-col justify-center text-center "
          }
        >
          Confusion Matrix View is currently not available.
        </div>
      )
    } else {
      return (
        <ConfusionMatrixBarCore
          height={height}
          width={width}
          data={confusionMatrixDataLoader.data}
        />
      )
    }
  }
}
