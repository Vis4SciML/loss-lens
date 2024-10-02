import React, { useEffect, useRef, useState } from "react"
import { useAtom } from "jotai"

import { selectedCheckPointIdListAtom } from "@/lib/store"

import LossLandscape from "./visualization/LossLandscapeModule"
import MergeTreeModule from "./visualization/MergeTreeModule"
import PersistenceBarcode from "./visualization/PersistenceBarcodeModule"

export default function LocalStructure() {
    const [selectedCheckPointIdList] = useAtom(selectedCheckPointIdListAtom)
    const containerRef = useRef<HTMLDivElement>(null)
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight,
                })
            }
        }

        updateDimensions()
        window.addEventListener("resize", updateDimensions)
        return () => window.removeEventListener("resize", updateDimensions)
    }, [])

    const renderContent = () => {
        if (selectedCheckPointIdList.length === 0) {
            return (
                <>
                    <div className="col-span-5 h-full p-1">
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a case study to start
                        </div>
                    </div>
                    <div className="col-span-5 h-full p-1">
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
                    <div className="col-span-5 h-full p-1" key={mId}>
                        <div className="flex h-full w-full items-center justify-center rounded border text-gray-500">
                            please select a checkpoint
                        </div>
                    </div>
                )
            }
            return (
                <div className="col-span-5 h-full" key={mId}>
                    <div className="relative grid h-full grid-cols-3">
                        <div className="col-span-1 h-full">
                            <LossLandscape
                                dimensions={dimensions}
                                checkpointId={checkPointId}
                            />
                        </div>
                        <div className="col-span-1 h-full">
                            <PersistenceBarcode
                                dimensions={dimensions}
                                checkpointId={checkPointId}
                            />
                        </div>
                        <div className="col-span-1 h-full">
                            <MergeTreeModule
                                dimensions={dimensions}
                                checkpointId={checkPointId}
                            />
                        </div>
                    </div>
                </div>
            )
        })
    }

    return (
        <div ref={containerRef} className="grid h-[calc(23vh)]">
            <div className="grid grid-cols-10">{renderContent()}</div>
        </div>
    )
}
