import { useAtom } from "jotai"

import {
    loadableSemiGlobalLocalStructureAtom,
    selectedCheckPointIdListAtom,
} from "@/lib/store"

import SemiGlobalLocalCore from "./SemiGlobalLocalCore"

export default function SemiGlobalLocalModule({ height, width }) {
    const [loader] = useAtom(loadableSemiGlobalLocalStructureAtom)
    const [selectedCheckPointIdList, setSelectedCheckPointIdList] = useAtom(
        selectedCheckPointIdListAtom
    )
    const updateCheckpointId = (index: number, newId: string) => {
        setSelectedCheckPointIdList({ index, value: newId })
    }
    const canvasHeight = height - 170
    const canvasWidth = width - 30

    if (loader.state === "hasError") {
        return <div>error</div>
    } else if (loader.state === "loading") {
        return <div>loading</div>
    } else {
        if (loader.data === null) {
            return (
                <div className="grid grid-cols-11 ">
                    <div className="col-span-1 flex h-full items-center justify-center  font-serif text-lg">
                        Global Structure
                    </div>
                    <div className="col-span-5  aspect-square  p-1 ">
                        <div className=" flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a case study to start
                        </div>
                    </div>
                    <div className="col-span-5  aspect-square  p-1">
                        <div className=" flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a case study to start
                        </div>
                    </div>
                </div>
            )
        } else {
            const modelList = loader.data.modelList

            const viewComponentList = modelList.map(
                (modelId: string, modelIdIndex: number) => {
                    return (
                        <div className="col-span-5 aspect-square p-1">
                            <SemiGlobalLocalCore
                                height={canvasHeight / 2}
                                width={canvasWidth}
                                data={loader.data}
                                selectedCheckPointIdList={
                                    selectedCheckPointIdList
                                }
                                updateSelectedModelIdModeId={updateCheckpointId}
                                modelId={modelId}
                                modelIdIndex={modelIdIndex}
                            />
                        </div>
                    )
                }
            )

            return (
                <div className="grid grid-cols-11 ">
                    <div className="col-span-1 flex h-full items-center justify-center  font-serif text-lg">
                        Global Structure
                    </div>
                    {viewComponentList}
                </div>
            )
        }
    }
}
