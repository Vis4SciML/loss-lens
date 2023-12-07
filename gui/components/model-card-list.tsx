import { useEffect } from "react"
import { useAtom } from "jotai"

import {
  loadModelMetaDataListAtom,
  loadModelUIListAtom,
  systemConfigAtom,
  updateModelUIListAtom,
} from "@/lib/losslensStore"
import { modelColor } from "@/styles/vis-color-scheme"

import { Label } from "./ui/label"
import { Separator } from "./ui/separator"

export default function ModelCardList() {
  const [systemConfig] = useAtom(systemConfigAtom)
  const [modelMetaDataList, fetchData] = useAtom(loadModelMetaDataListAtom)
  const [modelUIList, loadModelUIList] = useAtom(loadModelUIListAtom)
  const [, updateModelUIList] = useAtom(updateModelUIListAtom)

  useEffect(() => {
    if (systemConfig) {
      fetchData()
      loadModelUIList()
    }
  }, [systemConfig, fetchData, loadModelUIList])

  if (modelMetaDataList) {
    const modelCards = modelMetaDataList.map((model, index) => {
      const modelDescriptionUI = model.modelDescription
        .split(",")
        .map((d, i) => {
          if (i === model.modelDescription.split(",").length - 1) return d
          return d + ","
        })
      const datasetDescriptionUI = model.modelDatasetDescription
        .split(",")
        .map((d) => {
          return <div className="font-serif text-lg">{d}</div>
        })
      return (
        <div className="col-span-5 h-full p-2">
          <div className="flex justify-start">
            <div className="col-span-1 flex justify-end px-2 py-1">
              <svg width={20} height={20}>
                <circle
                  fill={modelColor[index]}
                  cx={10}
                  cy={10}
                  r={10}
                ></circle>
              </svg>
            </div>
            <div>
              <div className="font-serif text-xl font-semibold">
                {model.modelName}
              </div>
              <Label className="font-serif text-lg font-semibold">
                {" "}
                Model Info
              </Label>
              <div className="font-serif">{modelDescriptionUI}</div>
              {datasetDescriptionUI && (
                <>
                  <Label className="font-serif text-lg font-semibold">
                    Dataset:
                  </Label>
                  <div className="font-serif">{datasetDescriptionUI}</div>
                </>
              )}
            </div>
          </div>
        </div>
      )
    })
    return (
      <div className="grid h-auto grid-cols-11">
        <div className="col-span-1 flex h-full items-center justify-center font-serif text-lg"></div>
        {modelCards}
      </div>
    )
  }
  return (
    <div className="grid h-auto grid-cols-11">
      <div className="col-span-1 flex h-full items-center justify-center  font-serif text-lg"></div>
      <div className="col-span-5 h-full"></div>
      <div className="col-span-5 h-full"></div>
    </div>
  )
}
