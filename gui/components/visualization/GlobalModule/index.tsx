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
    showModelInfo,
    mcFilterRange,
}: {
    height: number
    width: number
    showPerformance: boolean
    showHessian: boolean
    showPerformanceLabels: boolean
    showModelInfo: boolean
    mcFilterRange: [number, number]
}) {
    const [loader] = useAtom(loadableSemiGlobalLocalStructureAtom)
    const [selectedCheckPointIdList, setSelectedCheckPointIdList] = useAtom(
        selectedCheckPointIdListAtom
    )
    const [modelMetaDataLoader] = useAtom(modelIDLoadableAtom)

    const updateCheckpointId = (index: number, newId: string) => {
        setSelectedCheckPointIdList({ index, value: newId })
    }

    const renderContent = () => {
        const placeholder = (
            <div className="flex">
                <div
                    className="flex-1 p-1"
                    style={{ height: height, width: width / 2 }}
                >
                    <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                        {loader.state === "hasError" ||
                        modelMetaDataLoader.state === "hasError"
                            ? "Error loading data"
                            : loader.state === "loading" ||
                                modelMetaDataLoader.state === "loading"
                              ? "Loading..."
                              : "please select a case study to start"}
                    </div>
                </div>
                <div
                    className="flex-1 p-1"
                    style={{ height: height, width: width / 2 }}
                >
                    <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                        {loader.state === "hasError" ||
                        modelMetaDataLoader.state === "hasError"
                            ? "Error loading data"
                            : loader.state === "loading" ||
                                modelMetaDataLoader.state === "loading"
                              ? "Loading..."
                              : "please select a case study to start"}
                    </div>
                </div>
            </div>
        )

        if (
            loader.state === "hasError" ||
            modelMetaDataLoader.state === "hasError" ||
            loader.state === "loading" ||
            modelMetaDataLoader.state === "loading" ||
            loader.data === null ||
            !modelMetaDataLoader.data
        ) {
            return placeholder
        } else {
            const modelList = loader.data.modelList
            const modelMetaDataList = modelMetaDataLoader.data.data

            return (
                <div className="flex">
                    {modelList.map((modelId: string, modelIdIndex: number) => {
                        const modelMetaData = modelMetaDataList[modelIdIndex]
                        return (
                            <div
                                key={modelId}
                                className="flex-1 p-1"
                                style={{ height: height, width: width / 2 }}
                            >
                                <GlobalCore
                                    height={height - 8}
                                    width={width / 2 - 8}
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
                                    showModelInfo={showModelInfo}
                                    mcFilterRange={mcFilterRange}
                                />
                            </div>
                        )
                    })}
                </div>
            )
        }
    }

    return (
        <div className="" style={{ height, width }}>
            {renderContent()}
        </div>
    )
}
