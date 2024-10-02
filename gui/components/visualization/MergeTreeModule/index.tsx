"use client"

import { useAtom } from "jotai"

import { fetchCheckpointMergeTreeDataAtomFamily } from "@/lib/store"

import MergeTreeCore from "./MergeTreeCore"

interface MergeTreeProps {
    dimensions: { width: number; height: number }
    checkpointId: string
}

export default function MergeTree({
    dimensions,
    checkpointId,
}: MergeTreeProps) {
    const [mergeTreeDataLoader] = useAtom(
        fetchCheckpointMergeTreeDataAtomFamily(checkpointId)
    )

    if (mergeTreeDataLoader.state === "hasError") {
        return <div>error</div>
    } else if (mergeTreeDataLoader.state === "loading") {
        return <div>loading</div>
    } else {
        if (mergeTreeDataLoader.data === null) {
            return (
                <div className="h-full w-full flex items-center justify-center text-gray-500">
                    Merge Tree is empty
                </div>
            )
        } else {
            return (
                <MergeTreeCore
                    dimensions={dimensions}
                    data={mergeTreeDataLoader.data}
                />
            )
        }
    }
}
