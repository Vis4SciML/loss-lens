import { useAtom } from "jotai"

import { modelIDLoadableAtom } from "@/lib/store"
import { modelColor } from "@/styles/vis-color-scheme"

import { Label } from "./ui/label"

export default function ModelCardList() {
  const [modelMetaDataLoader] = useAtom(modelIDLoadableAtom)

  if (modelMetaDataLoader.state === "hasError") {
    return <div>error</div>
  } else if (modelMetaDataLoader.state === "loading") {
    return <div>loading</div>
  } else {
    if (modelMetaDataLoader.data === null) {
      return <div className={" h-[900px] w-full text-center "}>Empty</div>
    } else {
      console.log(modelMetaDataLoader)
      const modelMetaDataList = modelMetaDataLoader.data
      if (!modelMetaDataList) {
        return <div className={" h-[900px] w-full text-center "}>Empty</div>
      }
      const modelCards = modelMetaDataList.data.map((model, index) => {
        const modelDescriptionUI = model.modelDescription
          .split(",")
          .map((d: string, i: number) => {
            if (i === model.modelDescription.split(",").length - 1) return d
            return d + ","
          })
        const datasetDescriptionUI = model.modelDatasetDescription
          .split(",")
          .map((d: string) => {
            return <div className="font-serif text-lg">{d}</div>
          })
        return (
          <div className="col-span-5 h-full p-2">
            <div className="flex justify-start">
              <div className="font-serif text-xl font-semibold">
                <span>{model.modelName}</span>
              </div>
              <div className="p-1">
                <svg width={20} height={20}>
                  <circle
                    fill={modelColor[index]}
                    cx={10}
                    cy={10}
                    r={10}
                  ></circle>
                </svg>
              </div>
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
        )
      })
      return (
        <div className="grid h-auto grid-cols-11">
          <div className="col-span-1 flex h-full items-center justify-center font-serif text-lg"></div>
          {modelCards}
        </div>
      )
    }
  }
}
