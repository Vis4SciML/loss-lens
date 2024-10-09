"use client"

import { useEffect, useState } from "react"
import dynamic from "next/dynamic"
import { atom, useAtom } from "jotai"

// Remove the import for react-range
// import { Range } from "react-range"

import { siteConfig } from "@/config/site"
import { systemConfigAtom } from "@/lib/store"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider" // Import your custom slider

import { Icons } from "@/components/icons"

const GlobalModuleNoSSR = dynamic(
    () => import("@/components/visualization/GlobalModule"),
    { ssr: false }
)

const LocalStructureNoSSR = dynamic(
    () => import("@/components/local-structure"),
    { ssr: false }
)

const ModelCardListNoSSR = dynamic(
    () => import("@/components/model-card-list"),
    { ssr: false }
)

type WindowDimensions = {
    width: number
    height: number
}

const useWindowDimensions = (): WindowDimensions => {
    const [windowDimensions, setWindowDimensions] = useState<WindowDimensions>({
        width: typeof window !== "undefined" ? window.innerWidth : 0,
        height: typeof window !== "undefined" ? window.innerHeight : 0,
    })

    useEffect(() => {
        const handleResize = (): void => {
            setWindowDimensions({
                width: window.innerWidth,
                height: window.innerHeight,
            })
        }

        window.addEventListener("resize", handleResize)
        return (): void => window.removeEventListener("resize", handleResize)
    }, [])

    return windowDimensions
}

const selectedCaseStudyUIAtom = atom<string | null>(null)

export default function IndexPage() {
    const { height, width } = useWindowDimensions()
    const [selectedCaseStudyUI, setSelectedCaseStudyUI] = useAtom(
        selectedCaseStudyUIAtom
    )
    const [systemConfigs, setSystemConfigs] = useAtom(systemConfigAtom)
    const [showPerformance, setShowPerformance] = useState(true)
    const [showHessian, setShowHessian] = useState(true)
    const [showPerformanceLabels, setShowPerformanceLabels] = useState(true)
    const [lossRange, setLossRange] = useState([0, 100])
    const [isControlPanelOpen, setIsControlPanelOpen] = useState(true)
    const [showModelInfo, setShowModelInfo] = useState(true)
    const [mcFilterRange, setMcFilterRange] = useState<[-100, 0]>([-100, 0]) // Update initial range

    const handleClick = () => {
        setSystemConfigs((prev) => ({
            ...prev,
            selectedCaseStudy: selectedCaseStudyUI,
        }))
    }

    const caseStudyItems = systemConfigs?.caseStudyList.map((key) => (
        <SelectItem key={key} className="w-60" value={key}>
            {systemConfigs.caseStudyLabels[key]}
        </SelectItem>
    ))

    const toggleControlPanel = () => {
        setIsControlPanelOpen(!isControlPanelOpen)
    }

    // Calculate dimensions for components
    const controlPanelWidth = isControlPanelOpen
        ? (width * 2) / 12
        : width * 0.05
    const mainContentWidth = width - controlPanelWidth
    const globalModuleHeight = height * 0.7
    const localStructureHeight = height * 0.3

    return (
        <div className="h-screen w-screen">
            <div className="grid h-full grid-cols-12">
                <div
                    className={`${
                        isControlPanelOpen ? "col-span-2" : "col-span-1"
                    } h-full`}
                >
                    <div className="h-full">
                        {isControlPanelOpen ? (
                            <div className="h-full w-full overflow-auto rounded-sm border border-gray-200 p-2">
                                <div className="w-full rounded-sm border border-gray-200 p-2">
                                    <div className="flex flex-col items-start">
                                        <div className="font-serif text-xl font-extrabold">
                                            {siteConfig.name}
                                        </div>
                                        <div className="w-full py-1 text-sm">
                                            <Select
                                                onValueChange={
                                                    setSelectedCaseStudyUI
                                                }
                                            >
                                                <SelectTrigger id="framework">
                                                    <SelectValue placeholder="Select Case Study" />
                                                </SelectTrigger>
                                                <SelectContent
                                                    className="w-40"
                                                    position="popper"
                                                >
                                                    {caseStudyItems}
                                                </SelectContent>
                                            </Select>
                                        </div>
                                        <div className="flex w-full justify-between">
                                            <Button
                                                size="xs"
                                                className="w-full rounded-sm"
                                                variant="outline"
                                                onClick={handleClick}
                                            >
                                                Apply
                                            </Button>
                                            <Button
                                                size="xs"
                                                variant="ghost"
                                                onClick={toggleControlPanel}
                                            >
                                                <Icons.chevronLeft className="h-5 w-5" />
                                            </Button>
                                        </div>
                                        <div className="w-full py-2 text-xs">
                                            <div className="mb-2 font-serif text-sm font-bold">
                                                Global View Settings
                                            </div>
                                            <div className="mt-2 flex items-center justify-between">
                                                <label
                                                    htmlFor="showPerformanceLabels"
                                                    className="mr-2"
                                                >
                                                    Show Performance Labels
                                                </label>
                                                <Checkbox
                                                    id="showPerformanceLabels"
                                                    checked={
                                                        showPerformanceLabels
                                                    }
                                                    onCheckedChange={(
                                                        checked
                                                    ) =>
                                                        setShowPerformanceLabels(
                                                            checked === true
                                                        )
                                                    }
                                                />
                                            </div>
                                            <div className="mt-2 flex items-center justify-between">
                                                <label
                                                    htmlFor="showPerformance"
                                                    className="mr-2"
                                                >
                                                    Show Performance
                                                </label>
                                                <Checkbox
                                                    id="showPerformance"
                                                    checked={showPerformance}
                                                    onCheckedChange={(
                                                        checked
                                                    ) =>
                                                        setShowPerformance(
                                                            checked === true
                                                        )
                                                    }
                                                />
                                            </div>
                                            <div className="mt-2 flex items-center justify-between">
                                                <label
                                                    htmlFor="showHessian"
                                                    className="mr-2"
                                                >
                                                    Show Hessian
                                                </label>
                                                <Checkbox
                                                    id="showHessian"
                                                    checked={showHessian}
                                                    onCheckedChange={(
                                                        checked
                                                    ) =>
                                                        setShowHessian(
                                                            checked === true
                                                        )
                                                    }
                                                />
                                            </div>
                                            <div className="mt-2 flex items-center justify-between">
                                                <label
                                                    htmlFor="showModelInfo"
                                                    className="mr-2"
                                                >
                                                    Show Model Info
                                                </label>
                                                <Checkbox
                                                    id="showModelInfo"
                                                    checked={showModelInfo}
                                                    onCheckedChange={(
                                                        checked
                                                    ) =>
                                                        setShowModelInfo(
                                                            checked === true
                                                        )
                                                    }
                                                />
                                            </div>
                                            <div className="mt-2">
                                                <label
                                                    htmlFor="mcFilter"
                                                    className="mr-2"
                                                >
                                                    Mode Connectivity Filter
                                                </label>
                                                <Slider
                                                    className="mt-2"
                                                    min={-100}
                                                    max={0}
                                                    step={1}
                                                    value={mcFilterRange} // Pass the value to the Slider
                                                    onValueChange={
                                                        setMcFilterRange
                                                    } // Use onValueChange to update the state
                                                />
                                                <div className="mt-1 flex justify-between text-xs">
                                                    <span>
                                                        {mcFilterRange[0]}
                                                    </span>
                                                    <span>
                                                        {mcFilterRange[1]}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <Button
                                size="sm"
                                variant="outline"
                                onClick={toggleControlPanel}
                                className="rounded-sm"
                            >
                                <Icons.chevronRight className="h-5 w-5" />
                            </Button>
                        )}
                    </div>
                </div>

                <div
                    className={`col-span-${
                        isControlPanelOpen ? "10" : "12"
                    } h-full`}
                >
                    <div className="flex h-full flex-col">
                        <div className="flex-grow">
                            <GlobalModuleNoSSR
                                height={globalModuleHeight}
                                width={mainContentWidth}
                                showPerformance={showPerformance}
                                showHessian={showHessian}
                                showPerformanceLabels={showPerformanceLabels}
                                showModelInfo={showModelInfo}
                                mcFilterRange={mcFilterRange}
                            />
                        </div>
                        <div className="flex-shrink-0">
                            <LocalStructureNoSSR
                                height={localStructureHeight}
                                width={mainContentWidth}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
