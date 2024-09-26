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
import LocalStructure from "@/components/local-structure"
import ModelCardList from "@/components/model-card-list"
import ModelComparisonPanel from "@/components/model-comparison-panel"
import SemiGlobalLocalModule from "@/components/visualization/SemiGlobalLocalModule"

const SemiGlobalLocalModuleNoSSR = dynamic(
  () => import("@/components/visualization/SemiGlobalLocalModule"),
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

const ModelComparisonPanelNoSSR = dynamic(
  () => import("@/components/model-comparison-panel"),
  { ssr: false }
)

type WindowDimentions = {
  width: number | undefined
  height: number | undefined
}

const useWindowDimensions = (): WindowDimentions => {
  const [windowDimensions, setWindowDimensions] = useState<WindowDimentions>({
    width: undefined,
    height: undefined,
  })
  useEffect(() => {
    function handleResize(): void {
      setWindowDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      })
    }
    handleResize()
    window.addEventListener("resize", handleResize)
    return (): void => window.removeEventListener("resize", handleResize)
  }, []) // Empty array ensures that effect is only run on mount

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
    setSystemConfigs({
      ...systemConfigs,
      selectedCaseStudy: selectedCaseStudyUI,
    })
  }

  const caseStudyItems = systemConfigs?.caseStudyList.map((key) => (
    <SelectItem className="w-60" value={key}>
      {systemConfigs.caseStudyLabels[key]}
    </SelectItem>
  ))

  return (
    <section className="p-4">
      <div className="flex flex-row justify-center">
        <div className="mr-4 hidden font-serif text-3xl font-extrabold sm:inline-block">
          {siteConfig.name}
        </div>
        <div className="w-40 pt-2">
          <Select
            onValueChange={(value: string) => {
              setSelectedCaseStudyUI(value)
            }}
          >
            <SelectTrigger id="framework">
              <SelectValue placeholder="Select Case Study" />
            </SelectTrigger>
            <SelectContent className="w-40" position="popper">
              {caseStudyItems}
            </SelectContent>
          </Select>
        </div>
        <div className="pt-2">
          <Button size={"xs"} variant="ghost" onClick={handleClick}>
            <Icons.play className="h-5 w-5" />
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-12">

        <div className="col-span-1 h-[calc(100vh-4rem)]"></div>
        <div className="col-span-10 h-[calc(100vh-4rem)]">
          <SemiGlobalLocalModuleNoSSR height={height} width={(width * 4) / 9} />
          <ModelCardListNoSSR />
          <LocalStructureNoSSR height={400} width={400} />
        </div>
        <div className="col-span-1 h-[calc(100vh-4rem)]"></div>
        {/* <ModelComparisonPanelNoSSR height={height} width={(width * 4) / 9} /> */}
      </div>
    </section>
  )
}
