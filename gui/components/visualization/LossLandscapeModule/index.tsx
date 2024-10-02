"use client"

import { useAtom } from "jotai"

import {
    fetchCheckpointLossLandscapeDataAtomFamily,
    modelIDLoadableAtom,
} from "@/lib/store"

import LossContourCore from "./LossContour"

interface LossLandscapeProps {
    dimensions: { width: number; height: number }
    checkpointId: string
}

export default function LossLandscape({
    dimensions,
    checkpointId,
}: LossLandscapeProps) {
    const [lossLandscapeDataLoader] = useAtom(
        fetchCheckpointLossLandscapeDataAtomFamily(checkpointId)
    )
    const [globalInfoLoader] = useAtom(modelIDLoadableAtom)

    if (
        lossLandscapeDataLoader.state === "hasError" ||
        globalInfoLoader.state === "hasError"
    ) {
        return <div>error</div>
    } else if (
        lossLandscapeDataLoader.state === "loading" ||
        globalInfoLoader.state === "loading"
    ) {
        return <div>loading</div>
    } else {
        if (
            lossLandscapeDataLoader.data === null ||
            globalInfoLoader.data === null
        ) {
            return (
                <div className="flex h-full w-full items-center justify-center text-gray-500">
                    LossContour is empty
                </div>
            )
        } else {
            return (
                <LossContourCore
                    dimensions={dimensions}
                    data={lossLandscapeDataLoader.data}
                    globalInfo={globalInfoLoader.data}
                />
            )
        }
    }
}
