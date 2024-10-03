import React, { useEffect, useRef, useState } from "react"
import { useAtom } from "jotai"

import { selectedCheckPointIdListAtom } from "@/lib/store"

import LossLandscape from "./visualization/LossLandscapeModule"
import MergeTreeModule from "./visualization/MergeTreeModule"
import PersistenceBarcode from "./visualization/PersistenceBarcodeModule"

export default function LocalStructure({
    height,
    width,
}: {
    height: number
    width: number
}) {
    const [selectedCheckPointIdList] = useAtom(selectedCheckPointIdListAtom)
    const containerRef = useRef<HTMLDivElement>(null)

    const renderContent = () => {
        if (selectedCheckPointIdList.length === 0) {
            return (
                <>
                    <div className="col-span-5 h-full">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a case study to start
                        </div>
                    </div>
                    <div className="col-span-5 h-full">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a case study to start
                        </div>
                    </div>
                </>
            )
        }

        return selectedCheckPointIdList.map((checkPointId, mId) => {
            if (checkPointId === "") {
                return (
                    <div className="col-span-5 h-full" key={mId}>
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a checkpoint
                        </div>
                    </div>
                )
            }
            return (
                <div className="col-span-5 h-full" key={mId}>
                    <div className="relative grid h-full grid-cols-3 px-1">
                        <div className="col-span-1 h-[calc(100%-1.5rem)] rounded-sm border">
                            <LossLandscape
                                height={height - 4}
                                width={
                                    (width - 8) /
                                    (3 * selectedCheckPointIdList.length)
                                }
                                checkpointId={checkPointId}
                            />
                        </div>
                        <div className="col-span-1 ml-1 h-[calc(100%-1.5rem)] rounded-sm border">
                            <PersistenceBarcode
                                height={height - 4}
                                width={
                                    (width - 8) /
                                    (3 * selectedCheckPointIdList.length)
                                }
                                checkpointId={checkPointId}
                            />
                        </div>
                        <div className="col-span-1 ml-1 h-[calc(100%-1.5rem)] rounded-sm border">
                            <MergeTreeModule
                                height={height - 4}
                                width={
                                    (width - 8) /
                                        (3 * selectedCheckPointIdList.length) -
                                    8
                                }
                                checkpointId={checkPointId}
                            />
                        </div>
                    </div>
                </div>
            )
        })
    }

    return (
        <div ref={containerRef} className="grid" style={{ height, width }}>
            <div className="grid grid-cols-10">{renderContent()}</div>
        </div>
    )
}
