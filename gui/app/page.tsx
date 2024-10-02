"use client"

import { useEffect, useState } from "react"
import dynamic from "next/dynamic"
import { atom, useAtom } from "jotai"

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
import { Slider } from "@/components/ui/slider"
import { Icons } from "@/components/icons"
import GlobalModule from "@/components/visualization/GlobalModule"

const SemiGlobalLocalModuleNoSSR = dynamic(
    () => import("@/components/visualization/SemiGlobalLocalModule"),
    { ssr: false }
)

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
    const [showPerformance, setShowPerformance] = useState(false)
    const [showHessian, setShowHessian] = useState(false)
    const [showPerformanceLabels, setShowPerformanceLabels] = useState(false)
    const [lossRange, setLossRange] = useState([0, 100])

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
    const [isControlPanelOpen, setIsControlPanelOpen] = useState(true)

    const toggleControlPanel = () => {
        setIsControlPanelOpen(!isControlPanelOpen)
    }
    return (
        <section className="p-4">
            <div className="grid grid-cols-12">
                <div className="col-span-2 h-[calc(100vh-3rem)]">
                    <div className="">
                        {isControlPanelOpen ? (
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
                                            {/* <Icons.play className="h-5 w-5" /> */}
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
                                        <div className="flex items-center justify-between">
                                            <label
                                                htmlFor="showPerformance"
                                                className="mr-2"
                                            >
                                                Show Performance
                                            </label>
                                            <Checkbox
                                                id="showPerformance"
                                                checked={showPerformance}
                                                onCheckedChange={(checked) =>
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
                                                onCheckedChange={(checked) =>
                                                    setShowHessian(
                                                        checked === true
                                                    )
                                                }
                                            />
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
                                                checked={showPerformanceLabels}
                                                onCheckedChange={(checked) =>
                                                    setShowPerformanceLabels(
                                                        checked === true
                                                    )
                                                }
                                            />
                                        </div>
                                        <div className="mt-4">
                                            <label
                                                htmlFor="lossRange"
                                                className="mb-2 block"
                                            >
                                                Loss Range
                                            </label>
                                            <Slider
                                                id="lossRange"
                                                value={lossRange}
                                                onValueChange={setLossRange}
                                                min={0}
                                                max={100}
                                                step={1}
                                            />
                                            <div className="mt-2 flex justify-between">
                                                <span>{lossRange[0]}</span>
                                                <span>{lossRange[1]}</span>
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

                <div className="col-span-10 h-[calc(100vh-3rem)]">
                    <GlobalModuleNoSSR height={800} width={1200} />
                    <LocalStructureNoSSR height={500} width={500} />
                </div>
            </div>
        </section>
    )
}
