"use client"

import * as React from "react"
import * as d3 from "d3"

import { LayerSimilarityData } from "@/types/losslens"
import { layerSimilarityColor } from "@/styles/vis-color-scheme"

interface LayerSimilarityCoreProp {
  data: LayerSimilarityData
  height: number
  width: number
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  wrapperRef: React.RefObject<HTMLDivElement>,
  data: LayerSimilarityData
) {
  console.log("render")
  const xLabel = data.modelX
  const yLabel = data.modelY
  const divElement = wrapperRef.current
  const width = divElement?.clientWidth || 0
  const height = divElement?.clientHeight || 0

  const margin = {
    top: 80,
    right: 85,
    bottom: 90,
    left: 85,
  }
  const h = height - margin.top - margin.bottom
  const w = width - margin.left - margin.right
  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
    .select("g")
    .attr("transform", `translate(${margin.left},${margin.top})`)

  const yScale = d3.scaleBand().domain(data.yLabels).range([h, 0])

  const xScale = d3
    .scaleBand()
    .range([0, yScale.bandwidth() * data.xLabels.length])
    .domain(data.xLabels)

  const cells = svg.selectAll(".cell").data(data.grid)

  const lowerBound = data.lowerBound
  const upperBound = data.upperBound
  const customizedColors = layerSimilarityColor.gridColor

  const domainSteps = customizedColors.map((_color, i) => {
    return (
      upperBound - (i * (upperBound - lowerBound)) / customizedColors.length
    )
  })

  const color = d3.scaleLinear().domain(domainSteps).range(customizedColors)

  let tooltip = d3
    .select("#tooltip")
    .style("position", "absolute")
    .style("visibility", "hidden")
    .style("pointer-events", "none")
    .style("padding", "10px")
    .style("background", "rgba(0, 0, 0, 0.6)")
    .style("color", "#fff")
    .style("border-radius", "4px")
    .style("font-size", "0.9em")

  cells
    .join("rect")
    .attr("class", "cell")
    .attr("x", (_d, i) => xScale(data.xLabels[i % data.xLabels.length]))
    .attr("y", (_d, i) =>
      yScale(data.yLabels[Math.floor(i / data.xLabels.length)])
    )
    .attr("width", xScale.bandwidth())
    .attr("height", xScale.bandwidth())
    .style("fill", (d) => color(d.value))
    .on("mouseover", function (_, d) {
      d3.select(this).raise().attr("stroke", "black").attr("stroke-width", 2)
      tooltip
        .style("visibility", "visible")
        .html(
          "<div> Similarity: " +
            d.value +
            "</div>" +
            "<div> Layer of " +
            xLabel +
            " </div>" +
            "<div> " +
            data.xLabels[d.xId] +
            " </div>" +
            "<div> Layer of " +
            yLabel +
            " </div>" +
            "<div>" +
            data.yLabels[d.yId] +
            " </div>"
        )
    })
    .on("mousemove", function (event) {
      tooltip
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px")
    })
    .on("mouseout", function () {
      d3.select(this).attr("stroke", "none")
      tooltip.style("visibility", "hidden")
    })

  const legend = svg.selectAll(".legend").data([1])
  let lg = svg
    .append("defs")
    .append("linearGradient")
    .attr("id", "layersim")
    .attr("x1", "100%")
    .attr("x2", "0%")
    .attr("y1", "0%")
    .attr("y2", "0%")

  lg.selectAll("stop")
    .data(customizedColors)
    .join("stop")
    .attr("offset", (_d, i) => (i * 100) / customizedColors.length + "%")
    .style("stop-color", (d) => d)
    .style("stop-opacity", 1)

  const leftOffset = 100
  legend
    .join("rect")
    .attr("class", "legend")
    .attr("y", h + 10)
    .attr("x", leftOffset)
    .attr("height", 10)
    .attr("width", w - leftOffset)
    .attr("fill", "url(#layersim)")

  const legendScale = d3
    .scaleLinear()
    .range([leftOffset, w])
    .domain([lowerBound, upperBound])

  const legendAxis = svg.selectAll(".legendAxis").data([data])

  legendAxis
    .join("g")
    .attr("class", "legendAxis")
    .attr("transform", `translate(0, ${h + 20})`)
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
    .attr("transform", `translate(0, 0)`)
    .call(
      d3.axisTop(xScale).tickFormat((d, i) => {
        if (i % 10 === 0) return i
        return ""
      })
    )
    .selectAll(".tick text")
    .attr("font-size", "0.9rem")
    .attr("class", "font-serif")
    .attr("fill", "#000")

  const yAxis = svg.selectAll(".yAxis").data([data])
  yAxis
    .join("g")
    .attr("class", "yAxis")
    .attr("transform", `translate(0, 0)`)
    .call(
      d3.axisLeft(yScale).tickFormat((d, i) => {
        if (i % 10 === 0) return i
        return ""
      })
    )
    .selectAll(".tick text")
    .attr("font-size", "0.9rem")
    .attr("class", "font-serif")
    .attr("fill", "#000")

  legendAxis
    .selectAll(".legendLabel")
    .data([1])
    .join("text")
    .attr("class", "legendLabel font-serif")
    .attr("x", 0)
    .attr("y", 0)
    .attr("font-size", "0.9rem")
    .attr("text-anchor", "start")
    .attr("fill", "#000")
    .text("CKA Similarity")

  // svg
  //   .selectAll(".figure-label")
  //   .data([1])
  //   .join("text")
  //   .attr("class", "figure-label font-serif")
  //   .attr("x", w / 2)
  //   .attr("y", h + 60)
  //   .attr("font-size", "1rem")
  //   .attr("font-weight", "semi-bold")
  //   .attr("text-anchor", "middle ")
  //   .attr("fill", "#000")
  //   .text("Layer Similarity View")

  svg
    .selectAll(".xLabel")
    .data([1])
    .join("text")
    .attr("class", "xLabel font-serif")
    .attr("x", w / 2)
    .attr("y", -40)
    .attr("font-size", "0.9rem")
    .attr("font-weight", "semi-bold")
    .attr("text-anchor", "middle ")
    .attr("fill", "#000")
    .text(xLabel + " " + data.checkPointX)

  svg
    .selectAll(".yLabel")
    .data([1])
    .join("text")
    .attr("class", "yLabel font-serif")
    .attr("x", -h / 2)
    .attr("y", -50)
    .attr("font-size", "0.9rem")
    .attr("font-weight", "semi-bold")
    .attr("text-anchor", "middle ")
    .attr("fill", "#000")
    .attr("transform", `rotate(-90)`)
    .text(yLabel + " " + data.checkPointY)
}

export default function LayerSimilarityCore({
  width,
  height,
  data,
}: LayerSimilarityCoreProp): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)
  const wrapperRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    render(svg, wrapperRef, data)
  }, [data, width, height])

  return (
    <div ref={wrapperRef} className="h-full w-full">
      <svg ref={svg}>
        <g></g>
      </svg>
      <div
        id="tooltip"
        style={{
          position: "absolute",
          visibility: "hidden",
          pointerEvents: "none",
          padding: "10px",
          background: "rgba(0, 0, 0, 0.6)",
          color: "#fff",
          borderRadius: "4px",
          fontSize: "0.9em",
        }}
      ></div>
    </div>
  )
}
