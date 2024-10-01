import { useAtom } from "jotai"

import {
    loadableSemiGlobalLocalStructureAtom,
    modelIDLoadableAtom,
    selectedCheckPointIdListAtom,
} from "@/lib/store"
import { modelColor } from "@/styles/vis-color-scheme"

import GlobalCore from "./GlobalCore"

export default function GlobalModule({
    height,
    width,
}: {
    height: number
    width: number
}) {
    const [loader] = useAtom(loadableSemiGlobalLocalStructureAtom)
    const [selectedCheckPointIdList, setSelectedCheckPointIdList] = useAtom(
        selectedCheckPointIdListAtom
    )
    const [modelMetaDataLoader] = useAtom(modelIDLoadableAtom)

    const updateCheckpointId = (index: number, newId: string) => {
        setSelectedCheckPointIdList({ index, value: newId })
    }

    const canvasHeight = height - 170
    const canvasWidth = width - 30

    if (
        loader.state === "hasError" ||
        modelMetaDataLoader.state === "hasError"
    ) {
        return <div>error</div>
    } else if (
        loader.state === "loading" ||
        modelMetaDataLoader.state === "loading"
    ) {
        return <div>loading</div>
    } else {
        if (loader.data === null || !modelMetaDataLoader.data) {
            return (
                <div className="grid grid-cols-11 ">
                    <div className="col-span-11 flex h-full items-center justify-center rounded border text-gray-500">
                        please select a case study to start
                    </div>
                </div>
            )
        } else {
            const modelList = loader.data.modelList
            const modelMetaDataList = modelMetaDataLoader.data.data

            const viewComponentList = modelList.map(
                (modelId: string, modelIdIndex: number) => {
                    const modelMetaData = modelMetaDataList[modelIdIndex]
                    return (
                        <div
                            key={modelId}
                            className="col-span-5 aspect-square p-1"
                        >
                            <GlobalCore
                                height={canvasHeight / 2}
                                width={canvasWidth}
                                data={loader.data}
                                selectedCheckPointIdList={
                                    selectedCheckPointIdList
                                }
                                updateSelectedModelIdModeId={updateCheckpointId}
                                modelId={modelId}
                                modelIdIndex={modelIdIndex}
                                modelMetaData={modelMetaData}
                            />
                        </div>
                    )
                }
            )

            return (
                <div className="grid grid-cols-11">
                    <div className="col-span-1 flex h-full items-center justify-center font-serif text-lg">
                        Global Structure
                    </div>
                    <div className="col-span-10">
                        <div className="grid grid-cols-10">
                            {viewComponentList}
                        </div>
                    </div>
                </div>
            )
        }
    }
}
