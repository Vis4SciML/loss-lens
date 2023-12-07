"use client"

import * as React from "react"
import * as d3 from "d3"

import { RegressionDifferenceData } from "@/types/losslens"
import {
  modelColor,
  regressionDifferenceColor,
} from "@/styles/vis-color-scheme"

interface RegressionDifferenceCoreProps {
  data: RegressionDifferenceData
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
  wrapperRef: React.RefObject<HTMLDivElement>,
  data: RegressionDifferenceData
) {
  const divElement = wrapperRef.current
  const width = divElement?.clientWidth
  const height = divElement?.clientHeight
  const margin = {
    top: 50,
    right: 10,
    bottom: 40,
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

  console.log("RegressionDifferenceCore.tsx")
  console.log(data)

  const dataA = data.grid["a"]
  const dataB = data.grid["b"]

  const mi = d3.min([
    d3.min(dataA, (d) => d3.min(d)),
    d3.min(dataB, (d) => d3.min(d)),
  ])
  const ma = d3.max([
    d3.max(dataA, (d) => d3.max(d)),
    d3.max(dataB, (d) => d3.max(d)),
  ])

  // grid[0] is a 2d array where each element has 2 elements

  const xScale = d3.scaleLinear().domain([mi, ma]).range([0, w])
  const yScale = d3.scaleLinear().domain([mi, ma]).range([h, 0])

  const grid = []
  for (let i = 0; i < dataA.length; i++) {
    grid.push({
      label: "a",
      value: dataA[i],
    })
    grid.push({
      label: "b",
      value: dataB[i],
    })
  }

  svg
    .selectAll(".difference")
    .data(grid)
    .join("line")
    .attr("class", "difference")
    .attr("x1", (d) => xScale(d.value[0]))
    .attr("y1", (d) => yScale(d.value[0]))
    .attr("x2", (d) => xScale(d.value[0]))
    .attr("y2", (d) => yScale(d.value[1]))
    .attr("stroke", (d) =>
      d.label === "a"
        ? regressionDifferenceColor.aColor
        : regressionDifferenceColor.bColor
    )
    .attr("stroke-width", 1)
    .attr("stroke-opacity", 0.1)

  svg
    .selectAll(".diag")
    .data([1])
    .join("line")
    .attr("class", "diag")
    .attr("x1", xScale(mi))
    .attr("y1", yScale(mi))
    .attr("x2", xScale(ma))
    .attr("y2", yScale(ma))
    .attr("stroke", "#000")
    .attr("stroke-width", 1)
    .attr("stroke-opacity", 0.5)

  const xAxis = svg
    .selectAll(".x-axis")
    .data([1])
    .join("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(7))

  xAxis
    .selectAll(".axisLabel")
    .data([0])
    .join("text")
    .attr("class", "axisLabel font-serif")
    .attr("x", w - 30)
    .attr("y", -10)
    .attr("text-anchor", "start")
    .attr("font-size", "0.9rem")
    .text("Label")
    .attr("fill", regressionDifferenceColor.textColor)

  const yAxis = svg
    .selectAll(".y-axis")
    .data([1])
    .join("g")
    .attr("class", "y-axis")
    .call(d3.axisLeft(yScale).ticks(7))

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
    .attr("fill", regressionDifferenceColor.textColor)

  xAxis
    .selectAll(".tick text")
    .attr("font-size", "0.9rem")
    .attr("class", "font-serif")
    .attr("text-anchor", "middle")

  yAxis
    .selectAll(".tick text")
    .attr("font-size", "0.9rem")
    .attr("class", "font-serif")

  const legendGroup = svg
    .selectAll("legendgroup")
    .data([0, 1])
    .join("g")
    .attr("class", "legendgroup")
    .attr("transform", (d, i) => {
      if (d === 0 || d === 1) {
        return `translate(${0},${h + 70 + i * 35})`
      } else {
        return `translate(${0 * 135},${h + 100})`
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
        return regressionDifferenceColor.aColor
      } else if (d === 1) {
        return regressionDifferenceColor.bColor
      }
    })
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
        return parts[0]
      } else if (d === 1) {
        return parts[1]
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

export default function RegressionDifferenceCore({
  width,
  height,
  data,
}: RegressionDifferenceCoreProps): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)

  const wrapperRef = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    if (!data) return
    const clonedData = JSON.parse(JSON.stringify(data))
    render(svg, wrapperRef, clonedData)
  }, [data, width, height])

  return (
    <div ref={wrapperRef} className="h-full w-full">
      <svg ref={svg}>
        <g></g>
      </svg>
    </div>
  )
}
