"use client"

import { useAtom } from "jotai"

import { loadableConfusionMatrixBarDataAtom } from "@/lib/store"
import Loader from "@/components/loader"

import ConfusionMatrixBarCore from "./ConfusionMatrixBarCore"

export default function ConfusionMatrixBarModule({ height, width }) {
  const [confusionMatrixDataLoader] = useAtom(
    loadableConfusionMatrixBarDataAtom
  )

  if (confusionMatrixDataLoader.state === "hasError") {
    return (
      <div
        className={"flex h-[550px] w-full flex-col justify-center text-center "}
      >
        Confusion Matrix View is currently not available.
      </div>
    )
  } else if (confusionMatrixDataLoader.state === "loading") {
    return (
      <div
        className={"flex h-[550px] w-full flex-col items-center justify-center"}
      >
        <Loader />
      </div>
    )
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
