"use client"

import { useAtom } from "jotai"

import { loadableRegressionDifferenceDataAtom } from "@/lib/store"

import RegressionDifferenceCore from "./RegressionDifferenceCore"

export default function RegressionDifferenceModule({ height, width }) {
  const [regressionDifferenceDataLoader] = useAtom(
    loadableRegressionDifferenceDataAtom
  )

  if (regressionDifferenceDataLoader.state === "hasError") {
    return <div>error</div>
  } else if (regressionDifferenceDataLoader.state === "loading") {
    return <div>loading</div>
  } else {
    if (regressionDifferenceDataLoader.data === null) {
      return (
        <div
          className={
            "flex h-[550px] w-full flex-col justify-center text-center "
          }
        >
          Regression Difference View is currently not available.
        </div>
      )
    } else {
      return (
        <RegressionDifferenceCore
          height={height}
          width={width}
          data={regressionDifferenceDataLoader.data}
        />
      )
    }
  }
}
