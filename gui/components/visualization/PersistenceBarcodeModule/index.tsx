"use client"

import { useAtom } from "jotai"

import { fetchCheckpointPersistenceBarcodeDataAtomFamily } from "@/lib/store"

import PersistenceBarcodeCore from "./PersistenceBarcodeCore"

interface PersistenceBarcodeProps {
    height: number
    width: number
    checkpointId: string
}

export default function PersistenceBarcode({
    height,
    width,
    checkpointId,
}: PersistenceBarcodeProps) {
    const [persistenceBarcodeDataLoader] = useAtom(
        fetchCheckpointPersistenceBarcodeDataAtomFamily(checkpointId)
    )

    if (persistenceBarcodeDataLoader.state === "hasError") {
        return <div>error</div>
    } else if (persistenceBarcodeDataLoader.state === "loading") {
        return <div>loading</div>
    } else {
        if (persistenceBarcodeDataLoader.data === null) {
            return (
                <div className="h-full w-full flex items-center justify-center text-gray-500">
                    Persistence Barcode is empty
                </div>
            )
        } else {
            return (
                <PersistenceBarcodeCore
                    height={height}
                    width={width}
                    data={persistenceBarcodeDataLoader.data}
                />
            )
        }
    }
}
