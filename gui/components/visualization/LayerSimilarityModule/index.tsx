"use client"

import { useEffect, useState } from "react"
import { useAtom } from "jotai"

import {
  loadLayerSimilarityDataAtom,
  selectedModeIdListAtom,
  XCheckBoxAtom,
  YCheckBoxAtom,
} from "@/lib/losslensStore"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Icons } from "@/components/icons"

import LayerSimilarityCore from "./LayerSimilarityCore"

export default function LayerSimilarityModule({
  height,
  width,
  modelIdModeIds,
}) {
  const [selectedModeIdList] = useAtom(selectedModeIdListAtom)
  const [data, fetchData] = useAtom(loadLayerSimilarityDataAtom)
  const [xCheckBoxItems, setXCheckBoxItems] = useAtom(XCheckBoxAtom)
  const [yCheckBoxItems, setYCheckBoxItems] = useAtom(YCheckBoxAtom)

  useEffect(() => {
    if (selectedModeIdList.length === 2) {
      fetchData()
    }
  }, [selectedModeIdList, fetchData])

  const toggleXCheckBoxItems = (index: number) => {
    const newXCheckBoxItems = [...xCheckBoxItems]
    newXCheckBoxItems[index] = !newXCheckBoxItems[index]
    setXCheckBoxItems(newXCheckBoxItems)
  }

  const toggleYCheckBoxItems = (index: number) => {
    const newYCheckBoxItems = [...yCheckBoxItems]
    newYCheckBoxItems[index] = !newYCheckBoxItems[index]
    setYCheckBoxItems(newYCheckBoxItems)
  }

  if (data) {
    const selectedXLabelDropDownMenuItems = xCheckBoxItems?.map(
      (isChecked, index) => (
        <DropdownMenuCheckboxItem
          key={data.xLabels[index]}
          checked={isChecked}
          onCheckedChange={() => toggleXCheckBoxItems(index)}
        >
          {data.xLabels[index]}
        </DropdownMenuCheckboxItem>
      )
    )

    const selectedYLabelDropDownMenuItems = yCheckBoxItems?.map(
      (isChecked, index) => (
        <DropdownMenuCheckboxItem
          key={data.yLabels[index]}
          checked={isChecked}
          onCheckedChange={() => toggleYCheckBoxItems(index)}
        >
          {data.yLabels[index]}
        </DropdownMenuCheckboxItem>
      )
    )

    // const aefae =
    //   {/* <div className="flex justify-start"> */}
    //     {/*   <DropdownMenu> */}
    //     {/*     <DropdownMenuTrigger asChild> */}
    //     {/*       <Button variant="link" className="font-serif text-black"> */}
    //     {/*         <Icons.listChecks className="h-5 w-5 mr-2" /> */}
    //     {/*         Layers of Mode A */}
    //     {/*       </Button> */}
    //     {/*     </DropdownMenuTrigger> */}
    //     {/*     <DropdownMenuContent className="h-56 w-56 overflow-scroll"> */}
    //     {/*       <DropdownMenuLabel>Appearance</DropdownMenuLabel> */}
    //     {/*       <DropdownMenuSeparator /> */}
    //     {/*       {selectedXLabelDropDownMenuItems} */}
    //     {/*     </DropdownMenuContent> */}
    //     {/*   </DropdownMenu> */}
    //     {/**/}
    //     {/*   <DropdownMenu> */}
    //     {/*     <DropdownMenuTrigger asChild> */}
    //     {/*       <Button variant="link" className="font-serif text-black"> */}
    //     {/*         <Icons.listChecks className="h-5 w-5 mr-2" /> */}
    //     {/*         Layers of Mode B */}
    //     {/*       </Button> */}
    //     {/*     </DropdownMenuTrigger> */}
    //     {/*     <DropdownMenuContent className="h-56 w-56 overflow-scroll"> */}
    //     {/*       <DropdownMenuLabel>Appearance</DropdownMenuLabel> */}
    //     {/*       <DropdownMenuSeparator /> */}
    //     {/*       {selectedYLabelDropDownMenuItems} */}
    //     {/*     </DropdownMenuContent> */}
    //     {/*   </DropdownMenu> */}
    //     {/* </div> */}

    return (
      <LayerSimilarityCore
        height={height}
        width={width}
        data={data}
        xCheckBoxItems={xCheckBoxItems}
        yCheckBoxItems={yCheckBoxItems}
      />
    )
  }

  return (
    <div
      className={"flex h-[550px] w-full flex-col justify-center text-center "}
    >
      Layer Similarity View is currently not available.
    </div>
  )
}
