"use client"

import { useAtom, useSetAtom } from "jotai"

import { siteConfig } from "@/config/site"
import { applySystemConfigAtom, systemConfigUIAtom } from "@/lib/losslensStore"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { MainNav } from "@/components/main-nav"

import { Label } from "./ui/label"

export function VISSiteHeader() {
  const applyConfig = useSetAtom(applySystemConfigAtom)
  const handleClick = () => {
    applyConfig()
  }

  const [systemConfigsUI, setSystemConfigsUI] = useAtom(systemConfigUIAtom)
  const caseStudyItems = systemConfigsUI?.caseStudyList.map((key) => (
    <SelectItem className="w-60" value={key}>
      {systemConfigsUI.caseStudyLabels[key]}
    </SelectItem>
  ))
  return (
    <header className="sticky top-0 z-40 w-full bg-background">
      <div className="flex px-4 h-12 items-center space-x-4 sm:justify-between sm:space-x-0">
        <MainNav items={siteConfig.mainNav} />
        <div className="flex flex-1 items-center justify-end space-x-4">
          <nav className="flex items-center space-x-1">
            <Label className="w-30">Case Study</Label>
            <div className="w-40">
              <Select
                onValueChange={(value: string) => {
                  const newSystemConfigs = Object.assign({}, systemConfigsUI, {
                    selectedCaseStudy: value,
                  })
                  setSystemConfigsUI(newSystemConfigs)
                }}
              >
                <SelectTrigger id="framework">
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent className="w-40" position="popper">
                  {caseStudyItems}
                </SelectContent>
              </Select>
            </div>
            <Button
              size={"xs"}
              className="
              w-20
              "
              onClick={handleClick}
            >
              Apply
            </Button>

            {/* <Link */}
            {/*   href={siteConfig.links.github} */}
            {/*   target="_blank" */}
            {/*   rel="noreferrer" */}
            {/* > */}
            {/*   <div */}
            {/*     className={buttonVariants({ */}
            {/*       size: "sm", */}
            {/*       variant: "ghost", */}
            {/*     })} */}
            {/*   > */}
            {/*     <Icons.gitHub className="h-5 w-5" /> */}
            {/*     <span className="sr-only">GitHub</span> */}
            {/*   </div> */}
            {/* </Link> */}
            {/* <Link */}
            {/*   href={siteConfig.links.twitter} */}
            {/*   target="_blank" */}
            {/*   rel="noreferrer" */}
            {/* > */}
            {/*   <div */}
            {/*     className={buttonVariants({ */}
            {/*       size: "sm", */}
            {/*       variant: "ghost", */}
            {/*     })} */}
            {/*   > */}
            {/*     <Icons.twitter className="h-5 w-5 fill-current" /> */}
            {/*     <span className="sr-only">Twitter</span> */}
            {/*   </div> */}
            {/* </Link> */}
            {/* <ThemeToggle /> */}
          </nav>
        </div>
      </div>
    </header>
  )
}
