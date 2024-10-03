"use client"
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
    showPerformance,
    showHessian,
    showPerformanceLabels,
}: {
    height: number
    width: number
    showPerformance: boolean
    showHessian: boolean
    showPerformanceLabels: boolean
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

    const renderContent = () => {
        if (
            loader.state === "hasError" ||
            modelMetaDataLoader.state === "hasError"
        ) {
            return (
                <div className="col-span-10 grid grid-cols-2">
                    <div className="col-span-1 aspect-square p-1">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            Error loading data
                        </div>
                    </div>
                    <div className="col-span-1 aspect-square p-1">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            Error loading data
                        </div>
                    </div>
                </div>
            )
        } else if (
            loader.state === "loading" ||
            modelMetaDataLoader.state === "loading"
        ) {
            return (
                <div className="col-span-10 grid grid-cols-2">
                    <div className="col-span-1 aspect-square p-1">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            Loading...
                        </div>
                    </div>
                    <div className="col-span-1 aspect-square p-1">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            Loading...
                        </div>
                    </div>
                </div>
            )
        } else if (loader.data === null || !modelMetaDataLoader.data) {
            return (
                <div className="col-span-10 grid grid-cols-2">
                    <div className="col-span-1 aspect-square p-1">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a case study to start
                        </div>
                    </div>
                    <div className="col-span-1 aspect-square p-1">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a case study to start
                        </div>
                    </div>
                </div>
            )
        } else {
            const modelList = loader.data.modelList
            const modelMetaDataList = modelMetaDataLoader.data.data

            return (
                <div className="col-span-10 grid grid-cols-2">
                    {modelList.map((modelId: string, modelIdIndex: number) => {
                        const modelMetaData = modelMetaDataList[modelIdIndex]
                        return (
                            <div
                                key={modelId}
                                className="col-span-1 aspect-square px-1"
                            >
                                <GlobalCore
                                    height={canvasHeight / 2.1}
                                    width={canvasWidth / 2.1}
                                    data={loader.data}
                                    selectedCheckPointIdList={
                                        selectedCheckPointIdList
                                    }
                                    updateSelectedModelIdModeId={
                                        updateCheckpointId
                                    }
                                    modelId={modelId}
                                    modelIdIndex={modelIdIndex}
                                    modelMetaData={modelMetaData}
                                    showPerformance={showPerformance}
                                    showHessian={showHessian}
                                    showPerformanceLabels={
                                        showPerformanceLabels
                                    }
                                />
                            </div>
                        )
                    })}
                </div>
            )
        }
    }

    return <div className="grid grid-cols-10">{renderContent()}</div>
}
