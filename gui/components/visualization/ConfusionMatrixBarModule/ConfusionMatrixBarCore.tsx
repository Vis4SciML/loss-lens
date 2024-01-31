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
    top: 60,
    right: 85,
    bottom: 150,
    left: 105,
  }
  const h = height - margin.top - margin.bottom
  const w = width - margin.left - margin.right
  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
    .select("g")
    .attr("transform", `translate(${margin.left},${margin.top})`)

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

  const yScale = d3
    .scaleBand()
    .range([h, 0])
    .domain(data.classesName)
    .padding(0.2)

  const xScale = d3
    .scaleLinear()
    .domain([
      // -d3.max(data.grid, (d) => d3.max([d.tn[0] + d.fn[0], d.tn[1] + d.fn[1]])),
      0,
      d3.max(data.grid, (d) => d3.max([d.tp[0] + d.fp[0], d.tp[1] + d.fp[1]])),
    ])
    .range([0, w])

  const subYScale = d3.scaleBand().range([0, yScale.bandwidth()]).domain([0, 1])

  svg
    .selectAll(".barGroup")
    .data(data.grid)
    .join("g")
    .attr("class", "barGroup")
    .attr("transform", (d, i) => `translate(0, ${yScale(data.classesName[i])})`)
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
    .attr("transform", (_d, i) => `translate(0, ${subYScale(i)})`)
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
    .attr("height", subYScale.bandwidth())
    .attr("width", (d, i) => {
      if (i === 0 || i === 1) {
        return xScale(d.stack)
      } else {
        return xScale(-d.stack)
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
    .on("mouseover", function (_, d) {
      tooltip
        .style("visibility", "visible")
        .html("<div> " + d.value + " </div>")
    })
    .on("mousemove", function (event) {
      tooltip
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px")
    })
    .on("mouseout", function () {
      tooltip.style("visibility", "hidden")
    })

  const yAxis = svg
    .selectAll(".yAxis")
    .data([0])
    .join("g")
    .attr("class", "yAxis")
    .attr("transform", `translate(0,${0})`)
    .call(d3.axisLeft(yScale))

  yAxis
    .selectAll(".axisLabel")
    .data([0])
    .join("text")
    .attr("class", "axisLabel font-serif")
    .attr("x", 0)
    .attr("y", -20)
    .attr("text-anchor", "start")
    .attr("font-size", "1rem")
    .text("Label")
    .attr("fill", confusionMatrixColor.textColor)

  const xAxis = svg
    .selectAll(".xAxis")
    .data([0])
    .join("g")
    .attr("class", "xAxis")
    .attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(xScale).tickFormat((d) => d3.format(".2s")(d)))

  xAxis
    .selectAll(".axisLabel")
    .data([0])
    .join("text")
    .attr("class", "axisLabel font-serif")
    .attr("x", w + 35)
    .attr("y", -10)
    .attr("text-anchor", "middle")
    .attr("font-size", "1rem")
    .text("Predicted")
    .attr("fill", confusionMatrixColor.textColor)

  xAxis
    .selectAll(".tick text")
    .attr("font-size", "1rem")
    .attr("class", "font-serif")
    .attr("text-anchor", "end")
    .attr("transform", `translate (0, 0 ) rotate(-45) `)

  yAxis
    .selectAll(".tick text")
    .attr("font-size", "1rem")
    .attr("class", "font-serif")

  const legendGroup = svg
    .selectAll("legendgroup")
    .data([0, 1, 2])
    .join("g")
    .attr("class", "legendgroup")
    .attr("transform", (d, i) => {
      if (d === 0 || d === 1 || d === 2) {
        return `translate(${0},${h + 70 + i * 30})`
      } else {
        return `translate(${0},${h + 20 + i * 30})`
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

  legendGroup
    .selectAll(".legendText")
    .data((d) => [d])
    .join("text")
    .attr("class", "legendText font-serif")
    .attr("x", 30)
    .attr("y", 15)
    .attr("font-size", "0.9rem")
    .attr("text-anchor", "start")
    .attr("fill", "#000")
    .text((d, i) => {
      if (d === 0) {
        return "TP " + data.modelY + " " + data.checkPointY
      } else if (d === 1) {
        return "TP " + data.modelX + " " + data.checkPointX
      } else {
        return "FP"
      }
    })

  // svg
  //   .selectAll(".figure-label")
  //   .data([1])
  //   .join("text")
  //   .attr("class", "figure-label font-serif")
  //   .attr("x", w / 2)
  //   .attr("y", h + 100)
  //   .attr("font-size", "1rem")
  //   .attr("font-weight", "semi-bold")
  //   .attr("text-anchor", "middle")
  //   .attr("fill", "#000")
  //   .text("Prediction Disparity View")
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
