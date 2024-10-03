"use client"

import { useAtom } from "jotai"

import {
    fetchCheckpointLossLandscapeDataAtomFamily,
    modelIDLoadableAtom,
} from "@/lib/store"

import LossContourCore from "./LossContour"

interface LossLandscapeProps {
    height: number
    width: number
    checkpointId: string
}

export default function LossLandscape({
    height,
    width,
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
        return (
            <div className="flex h-full w-full items-center justify-center text-gray-500">
                error
            </div>
        )
    } else if (
        lossLandscapeDataLoader.state === "loading" ||
        globalInfoLoader.state === "loading"
    ) {
        return (
            <div className="flex h-full w-full items-center justify-center text-gray-500">
                loading
            </div>
        )
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
                    height={height}
                    width={width}
                    data={lossLandscapeDataLoader.data}
                    globalInfo={globalInfoLoader.data}
                />
            )
        }
    }
}
