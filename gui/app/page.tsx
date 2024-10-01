"use client"

import { useEffect, useState } from "react"
import dynamic from "next/dynamic"
import { atom, useAtom } from "jotai"

import { siteConfig } from "@/config/site"
import { systemConfigAtom } from "@/lib/store"
import { Button } from "@/components/ui/button"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
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
            <div className="fixed left-4 top-4 z-50">
                {isControlPanelOpen ? (
                    <div className="rounded-lg bg-white p-4 shadow-md">
                        <div className="flex flex-col items-start">
                            <div className="mb-4 font-serif text-3xl font-extrabold">
                                {siteConfig.name}
                            </div>
                            <div className="mb-4 w-40">
                                <Select onValueChange={setSelectedCaseStudyUI}>
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
                                    variant="ghost"
                                    onClick={handleClick}
                                >
                                    <Icons.play className="h-5 w-5" />
                                </Button>
                                <Button
                                    size="xs"
                                    variant="ghost"
                                    onClick={toggleControlPanel}
                                >
                                    <Icons.chevronLeft className="h-5 w-5" />
                                </Button>
                            </div>
                        </div>
                    </div>
                ) : (
                    <Button
                        size="sm"
                        variant="outline"
                        onClick={toggleControlPanel}
                        className="rounded-full"
                    >
                        <Icons.chevronRight className="h-5 w-5" />
                    </Button>
                )}
            </div>
            <div className="grid grid-cols-12">
                <div className="col-span-1 h-[calc(100vh-4rem)]"></div>
                <div className="col-span-10 h-[calc(100vh-4rem)]">
                    <GlobalModuleNoSSR height={800} width={1200} />
                    <LocalStructureNoSSR height={400} width={400} />
                </div>
                <div className="col-span-1 h-[calc(100vh-4rem)]"></div>
            </div>
        </section>
    )
}
