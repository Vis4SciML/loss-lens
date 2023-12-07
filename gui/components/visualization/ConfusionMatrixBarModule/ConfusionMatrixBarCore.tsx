"use client"

import * as React from "react"
import * as d3 from "d3"

import { ConfusionMaterixBarData } from "@/types/losslens"
import { confusionMatrixColor } from "@/styles/vis-color-scheme"

interface ConfusionMatrixBarCoreProp {
  data: ConfusionMaterixBarData
  height: number
  width: number
}

function splitStringAtMiddleUnderscore(str) {
  const underscoreCount = str.split("_").length - 1

  if (underscoreCount < 2) {
    console.log("Not enough underscores to split the string.")
    return null
  }

  const middleUnderscoreIndex = Math.floor(underscoreCount / 2)
  const underscores = str.match(/_/g)
  let underscorePosition = -1

  for (let i = 0; i <= middleUnderscoreIndex; i++) {
    underscorePosition = str.indexOf("_", underscorePosition + 1)
  }

  if (underscorePosition !== -1) {
    const firstPart = str.slice(0, underscorePosition)
    const secondPart = str.slice(underscorePosition + 1)
    return [firstPart, secondPart]
  } else {
    console.log("Error: Unable to find middle underscore.")
    return null
  }
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  wraperRef: React.RefObject<HTMLDivElement>,
  data: ConfusionMaterixBarData
) {
  const divElement = wraperRef.current
  const width = divElement?.clientWidth
  const height = divElement?.clientHeight
  const margin = {
    top: 50,
    right: 40,
    bottom: 60,
    left: 40,
  }
  const h = height - margin.top - margin.bottom
  const w = width - margin.left - margin.right
  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
    .select("g")
    .attr("transform", `translate(${margin.left},${margin.top})`)

  const xScale = d3
    .scaleBand()
    .range([0, w])
    .domain(data.classesName)
    .padding(0.2)

  const yScale = d3
    .scaleLinear()
    .domain([
      // -d3.max(data.grid, (d) => d3.max([d.tn[0] + d.fn[0], d.tn[1] + d.fn[1]])),
      0,
      d3.max(data.grid, (d) => d3.max([d.tp[0] + d.fp[0], d.tp[1] + d.fp[1]])),
    ])
    .range([h, 0])

  const subXScale = d3.scaleBand().range([0, xScale.bandwidth()]).domain([0, 1])

  const groups = svg
    .selectAll(".barGroup")
    .data(data.grid)
    .join("g")
    .attr("class", "barGroup")
    .attr("transform", (d, i) => `translate(${xScale(data.classesName[i])},0)`)
    .selectAll(".stackedBarGroup")
    .data((d) => {
      return [
        [
          { value: d.fp[0], stack: d.fp[0] + d.tp[0], gid: 0 },
          { value: d.tp[0], stack: d.tp[0], gid: 0 },
          // { value: d.fn[0], stack: -d.fn[0] - d.tn[0], gid: 0 },
          // { value: d.tn[0], stack: -d.tn[0], gid: 0 },
        ],
        [
          { value: d.fp[1], stack: d.fp[1] + d.tp[1], gid: 1 },
          { value: d.tp[1], stack: d.tp[1], gid: 1 },
          // { value: d.fn[1], stack: -d.fn[1] - d.tn[1], gid: 1 },
          // { value: d.tn[1], stack: -d.tn[1], gid: 1 },
        ],
      ]
    })
    .join("g")
    .attr("class", "stackedBarGroup")
    .attr("transform", (_d, i) => `translate(${subXScale(i)},0)`)
    .selectAll(".stackedBar")
    .data((d) => d)
    .join("rect")
    .attr("class", "stackedBar")
    .attr("x", 0)
    .attr("y", (d, i) => {
      if (i === 0 || i === 1) {
        return yScale(d.stack)
      } else {
        return yScale(0)
      }
    })
    .attr("width", subXScale.bandwidth())
    .attr("height", (d, i) => {
      if (i === 0 || i === 1) {
        return yScale(0) - yScale(d.stack)
      } else {
        return yScale(0) - yScale(-d.stack)
      }
    })
    .attr("fill", (d, i) => {
      if (i === 0 || i === 2) {
        return confusionMatrixColor.secondaryColor
      } else if (i === 1) {
        if (d.gid === 0) {
          return confusionMatrixColor.color2
        } else {
          return confusionMatrixColor.color1
        }
      } else {
        if (d.gid === 0) {
          return confusionMatrixColor.color1
        } else {
          return confusionMatrixColor.color2
        }
      }
    })
    .attr("stroke", "none")

  const xAxis = svg
    .selectAll(".xAxis")
    .data([0])
    .join("g")
    .attr("class", "xAxis")
    .attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale))

  xAxis
    .selectAll(".axisLabel")
    .data([0])
    .join("text")
    .attr("class", "axisLabel font-serif")
    .attr("x", w + 5)
    .attr("y", 5)
    .attr("text-anchor", "start")
    .attr("font-size", "0.9rem")
    .text("Label")
    .attr("fill", confusionMatrixColor.textColor)

  const yAxis = svg
    .selectAll(".yAxis")
    .data([0])
    .join("g")
    .attr("class", "yAxis")
    .attr("transform", `translate(0,0)`)
    .call(d3.axisLeft(yScale).tickFormat((d) => d3.format(".2s")(d)))

  yAxis
    .selectAll(".axisLabel")
    .data([0])
    .join("text")
    .attr("class", "axisLabel font-serif")
    .attr("x", 0)
    .attr("y", -10)
    .attr("text-anchor", "middle")
    .attr("font-size", "0.9rem")
    .text("Predicted")
    .attr("fill", confusionMatrixColor.textColor)

  xAxis
    .selectAll(".tick text")
    .attr("font-size", "0.9rem")
    .attr("class", "font-serif")
    .attr("text-anchor", "end")
    .attr(
      "transform",
      `rotate(-90) translate (-10, -${xScale.bandwidth() / 2} ) `
    )

  yAxis
    .selectAll(".tick text")
    .attr("font-size", "0.9rem")
    .attr("class", "font-serif")

  const legendGroup = svg
    .selectAll("legendgroup")
    .data([0, 1, 2])
    .join("g")
    .attr("class", "legendgroup")
    .attr("transform", (d, i) => {
      if (d === 0 || d === 1 || d === 2) {
        return `translate(${0},${h + 70 + i * 35})`
      } else {
        return `translate(${0},${h + 100})`
      }
    })

  legendGroup
    .selectAll(".legendRect")
    .data((d) => [d])
    .join("rect")
    .attr("class", "legendRect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", 20)
    .attr("height", 20)
    .attr("fill", (d, i) => {
      if (d === 0) {
        return confusionMatrixColor.color2
      } else if (d === 1) {
        return confusionMatrixColor.color1
      } else if (d === 2) {
        return confusionMatrixColor.secondaryColor
      }
    })
  // There are multiple "_" in the string, and we only want the one in the middle
  // first find how many "_" in the string

  // find "_" in the middle and split the string into two parts

  const parts = splitStringAtMiddleUnderscore(data.modePairId)

  legendGroup
    .selectAll(".legendText")
    .data((d) => [d])
    .join("text")
    .attr("class", "legendText font-serif")
    .attr("x", 30)
    .attr("y", 15)
    .attr("font-size", "1rem")
    .attr("text-anchor", "start")
    .attr("fill", "#000")
    .text((d, i) => {
      if (d === 0) {
        return "TP " + parts[0]
      } else if (d === 1) {
        return "TP " + parts[1]
      } else {
        return "FP"
      }
    })

  svg
    .selectAll(".figure-label")
    .data([1])
    .join("text")
    .attr("class", "figure-label font-serif")
    .attr("x", w / 2)
    .attr("y", -30)
    .attr("font-size", "1rem")
    .attr("font-weight", "semi-bold")
    .attr("text-anchor", "middle")
    .attr("fill", "#000")
    .text("Prediction Disparity View")
}

export default function ConfusionMatrixBarCore({
  width,
  height,
  data,
}: ConfusionMatrixBarCoreProp): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)
  const wraperRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (!data) return
    const clonedData = JSON.parse(JSON.stringify(data))
    render(svg, wraperRef, clonedData)
  }, [data, width, height])

  return (
    <div ref={wraperRef} className="h-full w-full">
      <svg ref={svg}>
        <g></g>
      </svg>
    </div>
  )
}
