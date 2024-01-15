"use client"

import * as React from "react"
import * as d3 from "d3"

import { PersistenceBarcode } from "@/types/losslens"
import { persistenceBarcodeColor } from "@/styles/vis-color-scheme"

interface PersistenceBarcodeCoreProp {
  data: PersistenceBarcode
  height: number
  width: number
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  wrapperRef: React.RefObject<HTMLDivElement>,
  data: PersistenceBarcode
) {
  const divElement = wrapperRef.current
  const width = divElement?.clientWidth
  const height = divElement?.clientHeight
  const margin = {
    top: 25,
    right: 0,
    bottom: Math.abs(height - width) - 5,
    left: 20,
  }
  const h = height - margin.top - margin.bottom
  const w = width - margin.left - margin.right
  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
    .select("g")
    .attr("transform", `translate(${margin.left},${margin.top})`)

  let edges = []

  for (let i = 0; i < data.edges.length; i++) {
    const e = data.edges[i]
    if (i % 2 === 0) {
      edges.push({ x: e.y0, y0: e.y0, y1: e.y1 })
    } else {
      edges[edges.length - 1].y1 = e.y1
    }
  }

  if (!edges) {
    return
  }

  let zoom = d3.zoom().on("zoom", handleZoom)

  function handleZoom(e) {
    svg.attr("transform", e.transform)
  }

  d3.select(svgRef.current).call(zoom)

  const mi = d3.min(edges.map((e) => e.y0))
  const ma = d3.max(edges.map((e) => e.y1))
  const xScale = d3.scaleLinear().range([0, w]).domain([mi, ma])

  const yScale = d3.scaleLinear().domain([mi, ma]).range([h, 0])

  svg
    .selectAll(".persistenceline")
    .data(edges)
    .join("line")
    .attr("class", "persistenceline")
    .attr("class", data.id)
    .attr("x1", (d) => xScale(d.x))
    .attr("y1", (d) => yScale(d.y0))
    .attr("x2", (d) => xScale(d.x))
    .attr("y2", (d) => yScale(d.y1))
    .attr("stroke", persistenceBarcodeColor.strokeColor)
    .attr("stroke-width", 1)

  svg
    .selectAll(".diag")
    .data([1])
    .join("line")
    .attr("class", "diag")
    .attr("x1", (d) => xScale(mi))
    .attr("y1", (d) => yScale(mi))
    .attr("x2", (d) => xScale(ma))
    .attr("y2", (d) => yScale(ma))
    .attr("stroke", persistenceBarcodeColor.strokeColor)
    .attr("stroke-width", 1)

  const xGroupBase1 = svg.selectAll(".x-axis1").data([1])

  const xGroup1 = xGroupBase1.join("g").attr("class", "x-axis1")

  xGroup1
    .call(d3.axisBottom(xScale).tickFormat(() => ""))
    .attr("transform", `translate(0, ${h})`)

  const xGroupBase2 = svg.selectAll(".xlabel").data([1])

  xGroupBase2
    .join("text")
    .text("Death")
    .attr("class", "xlabel font-serif")
    .attr("text-anchor", "middle")
    .attr("transform", `rotate(90, ${w - 10}, ${h / 2 - 10})`)
    .attr("x", w + 5)
    .attr("y", h / 2 - 5)
    .style("font-size", "1rem")
    .style("fill", persistenceBarcodeColor.textColor)

  const yGroupBase1 = svg.selectAll(".y-axis1").data([1])

  const yGroup1 = yGroupBase1.join("g").attr("class", "y-axis1")

  yGroup1
    .call(d3.axisRight(yScale).tickFormat((d, i) => ""))
    .attr("transform", `translate(${w},0)`)

  const yGroupBase2 = svg.selectAll(".ylabel").data([1])

  yGroupBase2
    .join("text")
    .attr("class", "ylabel font-serif")
    .text("Birth")
    .attr("x", w / 2)
    .attr("y", h + 20)
    .attr("text-anchor", "middle")
    .style("font-size", "1rem")
    .style("fill", "#000")

  xGroup1.selectAll("path").attr("stroke", "#000")
  xGroup1.selectAll(".tick line").attr("display", "none")
  xGroup1.selectAll(".tick text").attr("display", "none")
  yGroup1.selectAll("path").attr("stroke", "#000")
  yGroup1.selectAll(".tick line").attr("display", "none")
  yGroup1.selectAll(".tick text").attr("display", "none")

  svg
    .selectAll(".figure-label")
    .data([1])
    .join("text")
    .attr("class", "figure-label font-serif")
    .attr("x", w / 2)
    .attr("y", h + 45)
    .attr("text-align", "center")
    .attr("font-size", "1rem")
    .attr("font-weight", "semi-bold")
    .attr("text-anchor", "middle ")
    .attr("fill", "#000")
    .text("Persistence Barcode [" + data.modeId + "]")
}

export default function PersistenceBarcodeCore({
  width,
  height,
  data,
}: PersistenceBarcodeCoreProp): React.JSX.Element {
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
