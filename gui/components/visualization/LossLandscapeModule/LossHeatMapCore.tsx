"use client"

import * as React from "react"
import * as d3 from "d3"

import { GlobalInfo, LossLandscape } from "@/types/losslens"

interface LossHeatMapCoreProp {
  data: LossLandscape
  height: number
  width: number
  globalInfo: GlobalInfo
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  data: LossLandscape,
  width: number,
  height: number,
  globalInfo: GlobalInfo
) {
  const margin = {
    top: 20,
    right: 10,
    bottom: Math.abs(height - width),
    left: 10,
  }
  const h = height - margin.top - margin.bottom
  const w = width - margin.left - margin.right

  // const upperBound = globalInfo.lossBounds.upperBound
  // const lowerBound = globalInfo.lossBounds.lowerBound

  const upperBound = d3.max(data.grid, (d) => d3.max(d))
  const lowerBound = d3.min(data.grid, (d) => d3.min(d))
  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
    .select("g")
    .attr("transform", `translate(${margin.left},${margin.top})`)

  const xScale = d3
    .scaleBand()
    .range([0, w])
    .domain(data.grid.map((_d, i) => i))

  const yScale = d3
    .scaleBand()
    .domain(data.grid.map((_d, i) => i))
    .range([h, 0])

  const rows = svg.selectAll(".row").data(data.grid)

  const customizedColors = [
    "#004477",
    "#305685",
    "#4d6994",
    "#677da3",
    "#8091b2",
    "#99a6c1",
    "#b2bcd0",
    "#cbd2e0",
    "#e5e8ef",
    "#ffffff",
  ]

  const domainSteps = customizedColors.map((_color, i) => {
    return (
      lowerBound + (i * (upperBound - lowerBound)) / customizedColors.length
    )
  })

  const color = d3.scaleLinear().domain(domainSteps).range(customizedColors)

  rows
    .join("g")
    .attr("class", "row")
    .attr("transform", (_d, i) => `translate(0, ${yScale(i)})`)

  const row = rows.selectAll(".cell").data((d) => d)

  row
    .join("rect")
    .attr("class", "cell")
    .attr("x", (_d, i) => xScale(i))
    .attr("y", 0)
    .attr("width", xScale.bandwidth())
    .attr("height", yScale.bandwidth())
    .style("fill", (d) => color(d))
  // .style("stroke", "none")
  // .style("stroke-width", 0.5)

  const legend = svg.selectAll(".legend").data([1])
  let lg = svg
    .append("defs")
    .append("linearGradient")
    .attr("id", "lossgrad")
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
    .attr("y", h + 5)
    .attr("x", 40)
    .attr("height", 7)
    .attr("width", w - 40)
    .attr("stroke", "#666")
    .attr("fill", "url(#lossgrad)")

  const legendScale = d3
    .scaleLinear()
    .range([0, w - 40])
    .domain([lowerBound, upperBound])

  const legendAxis = svg.selectAll(".legendAxis").data([data])

  legendAxis
    .join("g")
    .attr("class", "legendAxis")
    .attr("transform", `translate(40, ${h + 12})`)
    .call(d3.axisBottom(legendScale).ticks(5))

  // legendAxis.select(".domain").attr("display", "none")
  legendAxis
    .selectAll(".tick text")
    .attr("font-size", "10px")
    .attr("fill", "#666")
    .attr("text-anchor", "end")
    .attr("transform", "rotate(-30) translate(0, 0)")
  legendAxis.selectAll(".tick line").attr("stroke", "#666")

  legendAxis
    .selectAll(".legendLabel")
    .data([1])
    .join("text")
    .attr("class", "legendLabel")
    .attr("x", -25)
    .attr("y", 5)
    .attr("font-size", "12px")
    .attr("fill", "#666")
    .text("Loss")

  // const zoom = d3
  //   .zoom()
  //   .on("zoom", zoomed)
  //   .extent([
  //     [0, 0],
  //     [width, height],
  //   ])
  //   .scaleExtent([0.5, 8])
  //
  // svgbase.call(zoom)
  //
  // const zoomContainer = svgbase.select(".zoom-container")
  //
  // function zoomed(event) {
  //   zoomContainer.attr("transform", event.transform)
  // }

  svg
    .selectAll(".figure-label")
    .data([1])
    .join("text")
    .attr("class", "figure-label")
    .attr("x", 20)
    .attr("y", -10)
    .attr("font-size", "0.8rem")
    .attr("font-weight", "bold")
    .attr("text-anchor", "start ")
    .attr("fill", "#666")
    .text("Loss 2D HeatMap [Seed " + data.modeId + "]")
}

export default function LossHeatMapCore({
  width,
  height,
  data,
  globalInfo,
}: LossHeatMapCoreProp): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)

  React.useEffect(() => {
    if (!data) return
    const clonedData = JSON.parse(JSON.stringify(data))
    render(svg, clonedData, width, height, globalInfo)
  }, [data, width, height, globalInfo])

  return (
    <svg ref={svg}>
      <g></g>
      {/* <g className="zoom-container"> */}
      {/*   <g className="links"></g> */}
      {/*   <g className="nodes"></g> */}
      {/* </g> */}
    </svg>
  )
}
