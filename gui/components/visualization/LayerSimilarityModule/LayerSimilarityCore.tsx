"use client"

import * as React from "react"
import * as d3 from "d3"

import { LayerSimilarityData } from "@/types/losslens"
import { layerSimilarityColor } from "@/styles/vis-color-scheme"

interface LayerSimilarityCoreProp {
  data: LayerSimilarityData
  height: number
  width: number
  xCheckBoxItems: string[]
  yCheckBoxItems: string[]
}
function formatString(inputString: string, limit: number) {
  if (!inputString) {
    return ""
  }
  if (inputString.length > limit) {
    return "..." + inputString.slice(-limit)
  } else {
    return inputString
  }
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  wrapperRef: React.RefObject<HTMLDivElement>,
  data: LayerSimilarityData,
  xCheckBoxItems: string[],
  yCheckBoxItems: string[]
) {
  //get the max length of the label
  const maxLength = Math.max(
    ...data.xLabels.map((d) => d.length),
    ...data.yLabels.map((d) => d.length)
  )
  const divElement = wrapperRef.current
  const width = divElement?.clientWidth
  const height = divElement?.clientHeight

  const margin = {
    top: 30,
    right: width * 0.3,
    bottom: width * 0.3,
    left: 10,
  }
  const h = height - margin.top - margin.bottom
  const w = width - margin.left - margin.right
  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
    .select("g")
    .attr("transform", `translate(${margin.left},${margin.top})`)

  const filteredRowData = data.grid
    .map((d, i) => {
      return {
        values: d,
        filteredRowId: i,
      }
    })
    .filter((_d, i) => yCheckBoxItems[i] === true)

  const filteredColData = filteredRowData[0].values
    .map((d, i) => {
      return {
        filteredColId: i,
        value: d,
      }
    })
    .filter((_d, i) => xCheckBoxItems[i] === true)

  const yScale = d3
    .scaleBand()
    .domain(filteredRowData.map((d) => d.filteredRowId))
    .range([h, 0])

  const xScale = d3
    .scaleBand()
    .range([0, w])
    .domain(filteredColData.map((d) => d.filteredColId))

  const xLabels = data.xLabels.filter((_d, i) => xCheckBoxItems[i] === true)
  const yLabels = data.yLabels.filter((_d, i) => yCheckBoxItems[i] === true)

  const rows = svg.selectAll(".row").data(filteredRowData)

  const customizedColors = layerSimilarityColor.gridColor

  const upperBound = Math.max(
    ...filteredRowData.map((d) =>
      Math.max(...d.values.filter((_dd, j) => xCheckBoxItems[j] === true))
    )
  )
  const lowerBound = Math.min(
    ...filteredRowData.map((d) =>
      Math.min(...d.values.filter((_dd, j) => xCheckBoxItems[j] === true))
    )
  )

  const domainSteps = customizedColors.map((_color, i) => {
    return (
      upperBound - (i * (upperBound - lowerBound)) / customizedColors.length
    )
  })

  const color = d3.scaleLinear().domain(domainSteps).range(customizedColors)

  rows
    .join("g")
    .attr("class", "row")
    .attr("transform", (d) => `translate(0, ${yScale(d.filteredRowId)})`)

  const row = rows.selectAll(".cell").data((d) =>
    d.values
      .map((dd, i) => {
        return { value: dd, filteredColId: i }
      })
      .filter((dd) => xCheckBoxItems[dd.filteredColId] === true)
  )

  row
    .join("rect")
    .attr("class", "cell")
    .attr("x", (d) => xScale(d.filteredColId))
    .attr("y", 0)
    .attr("width", xScale.bandwidth())
    .attr("height", yScale.bandwidth())
    .style("fill", (d) => color(d.value))
    .style("stroke", "none")
    .style("stroke-width", 0.5)

  const legend = svg.selectAll(".legend").data([1])
  let lg = svg
    .append("defs")
    .append("linearGradient")
    .attr("id", "layersim")
    .attr("x1", "0%")
    .attr("x2", "100%")
    .attr("y1", "0%")
    .attr("y2", "0%")

  lg.selectAll("stop")
    .data(customizedColors)
    .join("stop")
    .attr("offset", (_d, i) => (i * 100) / customizedColors.length + "%")
    .style("stop-color", (d) => d)
    .style("stop-opacity", 1)

  legend
    .join("rect")
    .attr("class", "legend")
    .attr("y", height - 60)
    .attr("x", 0)
    .attr("height", 10)
    .attr("width", width)
    .attr("fill", "url(#layersim)")

  const legendScale = d3
    .scaleLinear()
    .range([0, width])
    .domain([lowerBound, upperBound])

  const legendAxis = svg.selectAll(".legendAxis").data([data])

  legendAxis
    .join("g")
    .attr("class", "legendAxis")
    .attr("transform", `translate(0, ${height - 50})`)
    .call(d3.axisBottom(legendScale))

  legendAxis.select(".domain").attr("display", "none")
  legendAxis
    .selectAll(".tick text")
    .attr("class", "font-serif")
    .attr("font-size", "0.8rem")
    .attr("fill", "#000")
  legendAxis.selectAll(".tick line").attr("stroke", "#000")

  const xAxis = svg.selectAll(".xAxis").data([data])

  xAxis
    .join("g")
    .attr("class", "xAxis")
    .attr("transform", `translate(0, ${h})`)
    .call(
      d3.axisBottom(xScale).tickFormat((d, i) => {
        if (xLabels.length < 20) {
          return formatString(xLabels[i], 100)
        } else if (xLabels.length < 50) {
          if (i % 3 === 0) {
            return formatString(xLabels[i], 100)
          } else {
            return ""
          }
        } else {
          if (i % 10 === 0) {
            return formatString(xLabels[i], 100)
          } else {
            return ""
          }
        }
      })
    )
    .selectAll(".tick text")
    .attr("font-size", "0.9rem")
    .attr("class", "font-serif")
    .attr("fill", "#000")
    .attr("text-anchor", "end")
    .attr("transform", `rotate(-90) translate(-10, -${xScale.bandwidth() / 2})`)
  xAxis.select(".domain").attr("display", "none")
  xAxis.selectAll(".tick line").attr("display", "none")

  const yAxis = svg.selectAll(".yAxis").data([data])
  yAxis
    .join("g")
    .attr("class", "yAxis")
    .attr("transform", `translate(${w}, 0)`)
    .call(
      d3.axisRight(yScale).tickFormat((d, i) => {
        if (yLabels.length < 20) {
          return formatString(yLabels[i], 100)
        } else if (yLabels.length < 50) {
          if (i % 3 === 0) {
            return formatString(yLabels[i], 100)
          } else {
            return ""
          }
        } else {
          if (i % 10 === 0) {
            return formatString(yLabels[i], 100)
          } else {
            return ""
          }
        }
      })
    )
    .selectAll(".tick text")
    .attr("font-size", "0.9rem")
    .attr("class", "font-serif")
    .attr("fill", "#000")
  yAxis.select(".domain").attr("display", "none")
  yAxis.selectAll(".tick line").attr("display", "none")

  legendAxis
    .selectAll(".legendLabel")
    .data([1])
    .join("text")
    .attr("class", "legendLabel font-serif")
    .attr("x", width - 60)
    .attr("y", -20)
    .attr("font-size", "0.9rem")
    .attr("fill", "#000")
    .text("CKA Similarity")

  svg
    .selectAll(".figure-label")
    .data([1])
    .join("text")
    .attr("class", "figure-label font-serif")
    .attr("x", w / 2)
    .attr("y", -10)
    .attr("font-size", "1rem")
    .attr("font-weight", "semi-bold")
    .attr("text-anchor", "start ")
    .attr("fill", "#000")
    .text("Layer Similarity View")
}

export default function LayerSimilarityCore({
  width,
  height,
  data,
  xCheckBoxItems,
  yCheckBoxItems,
}: LayerSimilarityCoreProp): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)
  const wrapperRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (!data || xCheckBoxItems.length === 0 || yCheckBoxItems.length === 0)
      return
    // const clonedData = JSON.parse(JSON.stringify(data))
    render(svg, wrapperRef, data, xCheckBoxItems, yCheckBoxItems)
  }, [data, width, height, xCheckBoxItems, yCheckBoxItems])

  return (
    <div ref={wrapperRef} className="h-full w-full">
      <svg ref={svg}>
        <g></g>
      </svg>
    </div>
  )
}
