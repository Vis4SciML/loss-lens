"use client"

import * as React from "react"
import * as d3 from "d3"

import { GlobalInfo, LossLandscape } from "@/types/losslens"
import { lossContourColor } from "@/styles/vis-color-scheme"

interface LossContourCoreProp {
  data: LossLandscape | Promise<LossLandscape> | null
  height: number
  width: number
  globalInfo: GlobalInfo
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  wraperRef: React.RefObject<HTMLDivElement>,
  data: LossLandscape,
  globalInfo: GlobalInfo
) {
  const divElement = wraperRef.current
  const width = divElement?.clientWidth ?? 300
  const height = divElement?.clientHeight ?? 300

  const margin = {
    top: 30,
    right: 0,
    bottom: Math.abs(height - width) - 5,
    left: 20,
  }
  const h = height - margin.top - margin.bottom
  const w = width - margin.left - margin.right

  const upperBound = globalInfo.lossBounds.upperBound
  const lowerBound = globalInfo.lossBounds.lowerBound

  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
    .select("g")
    .attr("transform", `translate(${margin.left},${margin.top})`)

  const lengthOfGrid = data.grid.length
  const q = w / lengthOfGrid

  // const x0 = -q / 2
  // const x1 = w + 28 + q
  // const y0 = -q / 2
  // const y1 = h + q
  // const n = Math.ceil((x1 - x0) / q)
  // const m = Math.ceil((y1 - y0) / q)

  const thresholdArray = []
  for (let i = 0; i < 40; i++) {
    const threshold = lowerBound + (i / 39) * (upperBound - lowerBound)
    thresholdArray.push(threshold)
  }

  const customizedColors = lossContourColor.gridColor

  const domainSteps = customizedColors.map((_color, i) => {
    return (
      upperBound - (i * (upperBound - lowerBound)) / customizedColors.length
    )
  })

  const color = d3
    .scaleLinear()
    .domain(domainSteps)
    .range(customizedColors.reverse())

  // const color = d3.scaleSequentialLog(
  //   d3.extent(thresholdArray),
  //   d3.interpolateMagma
  // )
  const gridX = -q
  const gridY = -q
  const gridK = q

  const transform = ({ type, value, coordinates }) => {
    return {
      type,
      value,
      coordinates: coordinates.map((rings) => {
        return rings.map((points) => {
          return points.map(([x, y]) => [gridX + gridK * x, gridY + gridK * y])
        })
      }),
    }
  }

  const contours = d3
    .contours()
    .size([40, 40])
    .thresholds(thresholdArray)(data.grid.flat())
    .map(transform)

  svg
    .selectAll("path")
    .data(contours)
    .join("path")
    .attr("fill", (d) => color(d.value))
    .attr("stroke", "#000")
    .attr("stroke-width", 0.5)
    .attr("d", d3.geoPath())

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
    .data(customizedColors.reverse())
    .join("stop")
    .attr("offset", (_d, i) => (i * 100) / customizedColors.length + "%")
    .style("stop-color", (d) => d)
    .style("stop-opacity", 1)

  legend
    .join("rect")
    .attr("class", "legend")
    .attr("y", h + 3)
    .attr("x", 40)
    .attr("height", 7)
    .attr("width", w - 45)
    .attr("stroke", "#666")
    .attr("fill", "url(#lossgrad)")

  const legendScale = d3
    .scaleLinear()
    .range([0, w - 45])
    .domain([lowerBound, upperBound])

  const legendAxis = svg.selectAll(".legendAxis").data([data])

  legendAxis
    .join("g")
    .attr("class", "legendAxis")
    .attr("transform", `translate(40, ${h + 10})`)
    .call(d3.axisBottom(legendScale).ticks(4).tickFormat(d3.format(".2s")))

  // legendAxis.select(".domain").attr("display", "none")
  legendAxis
    .selectAll(".tick text")
    .attr("font-size", "0.8rem")
    .attr("fill", "#000")
    .classed("font-serif", true)
    .attr("text-anchor", "middle")
  legendAxis.selectAll(".tick line").attr("stroke", "#000")

  legendAxis
    .selectAll(".legendLabel")
    .data([1])
    .join("text")
    .attr("class", "legendLabel font-serif")
    .attr("x", -25)
    .attr("y", 5)
    .attr("font-size", "1rem")
    .attr("font-weight", "semibold")
    .attr("fill", "#000")
    .text("Loss")

  svg
    .selectAll(".figure-label")
    .data([1])
    .join("text")
    .attr("class", "figure-label font-serif")
    .attr("x", w / 2)
    .attr("y", h + 45)
    .attr("font-size", "1rem")
    .attr("font-weight", "semi-bold")
    .attr("text-anchor", "middle")
    .text("Loss Contour [" + data.modeId + "]")
}

export default function LossContourCore({
  width,
  height,
  data,
  globalInfo,
}: LossContourCoreProp): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)
  const wraperRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    // Function to update the D3 chart based on the div dimensions
    const updateChart = () => {
      // Get the div dimensions using the ref
      const divElement = wraperRef.current
      const width = divElement?.clientWidth
      const height = divElement?.clientHeight

      console.log("updateChart")
      console.log(width)
      console.log(height)

      // Use the dimensions to render or update your D3 chart
      // Example: Update D3 chart with new dimensions
      const svgE = d3.select(svg.current)
      svgE.attr("width", width).attr("height", height)

      // Your D3 rendering logic here...
    }

    // Call the updateChart function initially and on window resize
    updateChart()
    window.addEventListener("resize", updateChart)

    // Clean up the event listener on component unmount
    return () => {
      window.removeEventListener("resize", updateChart)
    }
  }, [])

  React.useEffect(() => {
    render(svg, wraperRef, data, globalInfo)
  }, [data, width, height, globalInfo])

  return (
    <div ref={wraperRef} className="h-full w-full">
      <svg ref={svg}>
        <g></g>
      </svg>
    </div>
  )
}
