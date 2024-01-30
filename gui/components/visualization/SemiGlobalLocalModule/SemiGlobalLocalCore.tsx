"use client"

import * as React from "react"
import * as d3 from "d3"

import {
  ModeConnectivityLink,
  ModeNode,
  SemiGlobalLocalStructure,
} from "@/types/losslens"
import { roundToPercentage } from "@/lib/utils"
import {
  modelColor,
  semiGlobalLocalSturctureColor,
} from "@/styles/vis-color-scheme"

interface SemiGlobalLocalCoreProp {
  data: SemiGlobalLocalStructure
  height: number
  width: number
  selectedCheckPointIdList: string[]
  updateSelectedModelIdModeId: (index: number, id: string) => void
  modelId: string
  modelIdIndex: number
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  wraperRef: React.RefObject<HTMLDivElement>,
  data: SemiGlobalLocalStructure,
  selectedCheckPointIdList: string[],
  updateSelectedModelIdModeId: (index: number, id: string) => void,
  modelId: string,
  modelIdIndex: number
) {
  const divElement = wraperRef.current
  const width = divElement?.clientWidth || 0
  const height = divElement?.clientHeight || 0
  const svgbase = d3.select(svgRef.current)
  const margin = {
    top: 10,
    right: 10,
    bottom: 10,
    left: 10,
  }
  const h = height - margin.top - margin.bottom
  const w = width - margin.left - margin.right

  svgbase.attr("width", width).attr("height", height)
  const wholeNodes = data.nodes
  const wholeLinks = data.links
  const modelList = data.modelList

  const xScale = d3
    .scaleLinear()
    .domain(d3.extent(wholeNodes.map((node) => node.x)))
    .range([0, w])

  const yScale = d3
    .scaleLinear()
    .domain(d3.extent(wholeNodes.map((node) => node.y)))
    .range([0, h])

  const linkThickness = d3
    .scaleLinear()
    .domain(d3.extent(wholeLinks.map((link) => link.weight)))
    .range([2, 20])

  const linkCurvature = d3
    .scaleLinear()
    .domain(d3.extent(wholeLinks.map((link) => link.weight)))
    .range([600, 1])

  const linkSmoothness = d3
    .scaleLinear()
    .domain(d3.extent(wholeLinks.map((link) => link.weight)))
    .range([1.0, 0.1])

  const modelColorMap = {}
  modelList.forEach((model, index) => {
    modelColorMap[model] = modelColor[index]
  })

  // modelUIList.forEach((modelUI) => {
  //   const modelId = modelUI.modelId
  //   const modelNumberOfModes = modelUI.selectedNumberOfModes
  //   const filteredNodes = wholeNodes
  //     .filter((node) => node.modelId === modelId)
  //     .filter((_node, index) => index < modelNumberOfModes)
  //   filteredNodes.forEach((node) => {
  //     filteredNodeList.add(node.modeId)
  //   })
  // })

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

  const links = wholeLinks.filter(
    (link) => link.source.modelId === modelId && link.target.modelId === modelId
  )

  const nodes = wholeNodes.filter((node) => node.modelId === modelId)

  function positionNode(d: ModeNode) {
    return "translate(" + xScale(d.x) + "," + yScale(d.y) + ")"
  }

  function positionLinkSmooth(d: ModeConnectivityLink) {
    let offset = linkCurvature(d.weight)

    const x1 = xScale(d.source.x)
    const y1 = yScale(d.source.y)
    const x2 = xScale(d.target.x)
    const y2 = yScale(d.target.y)

    const midpoint_x = (x1 + x2) / 2
    const midpoint_y = (y1 + y2) / 2

    const dx = x2 - x1
    const dy = y2 - y1

    let normalise = Math.sqrt(dx * dx + dy * dy)

    let offSetX = midpoint_x + offset * (dy / normalise)
    let offSetY = midpoint_y - offset * (dx / normalise)

    return (
      "M" + x1 + "," + y1 + "S" + offSetX + "," + offSetY + " " + x2 + "," + y2
    )
  }

  function positionLinkRough(d: ModeConnectivityLink) {
    let offset = 40

    const x1 = xScale(d.source.x)
    const y1 = yScale(d.source.y)
    const x2 = xScale(d.target.x)
    const y2 = yScale(d.target.y)

    const dx = x2 - x1
    const dy = y2 - y1

    const angle = Math.atan2(dy, dx)

    const numControlPoints = linkSmoothness(d.weight)

    // Build the path data string
    let pathData = `M${x1},${y1}`
    // pathData += ` C`

    // Calculate the control points and add them to the path data
    for (let i = 1; i <= numControlPoints; i++) {
      const t = i / (numControlPoints + 1)
      const cx1 = x1 + t * dx - offset * Math.sin(angle)
      const cy1 = y1 + t * dy + offset * Math.cos(angle)
      const cx2 = x1 + t * dx + offset * Math.sin(angle)
      const cy2 = y1 + t * dy - offset * Math.cos(angle)
      const x = x1 + t * dx
      const y = y1 + t * dy

      // Add the cubic Bezier curve to the path data
      pathData += ` C${cx1},${cy1} ${cx2},${cy2} ${x},${y}`
    }

    return pathData
  }

  function generateCurvePath(edge) {
    // Apply scaling to coordinates
    const scaledSourceX = xScale(edge.source.x)
    const scaledSourceY = yScale(edge.source.y)
    const scaledTargetX = xScale(edge.target.x)
    const scaledTargetY = yScale(edge.target.y)

    // Calculate midpoint
    const midX = (scaledSourceX + scaledTargetX) / 2
    const midY = (scaledSourceY + scaledTargetY) / 2

    // Calculate distance between points
    const dx = scaledTargetX - scaledSourceX
    const dy = scaledTargetY - scaledSourceY

    // Length of perpendicular offset
    const offsetLength =
      Math.sqrt(dx * dx + dy * dy) * linkSmoothness(edge.weight)

    // Calculate perpendicular vector
    const offsetX = -dy * (offsetLength / Math.sqrt(dx * dx + dy * dy))
    const offsetY = dx * (offsetLength / Math.sqrt(dx * dx + dy * dy))

    // Control point
    const ctrlX = midX + offsetX
    const ctrlY = midY + offsetY

    // Construct the path string
    return `M${scaledSourceX},${scaledSourceY} Q${ctrlX},${ctrlY} ${scaledTargetX},${scaledTargetY}`
  }

  function generateSpringPath(edge) {
    // Base number of zigzags
    const baseZigzags = 0
    // Calculate total number of zigzags based on edge value
    const totalZigzags =
      baseZigzags + Math.floor(linkSmoothness(edge.weight) * 10)

    // Scale the source and target coordinates
    const sourceX = xScale(edge.source.x)
    const sourceY = yScale(edge.source.y)
    const targetX = xScale(edge.target.x)
    const targetY = yScale(edge.target.y)

    // Calculate the vector from source to target
    const dx = targetX - sourceX
    const dy = targetY - sourceY
    const length = Math.sqrt(dx * dx + dy * dy)

    // Initialize the path data string
    let pathData = `M${sourceX},${sourceY}`

    // Add each segment of the spring
    for (let i = 0; i < totalZigzags; i++) {
      // Calculate the control point for this segment
      const segmentLength = length / totalZigzags
      const angle =
        Math.atan2(dy, dx) + (i % 2 === 0 ? Math.PI / 2 : -Math.PI / 2)
      const offsetLength = segmentLength * linkSmoothness(edge.weight) // Adjust this to change the amplitude of the zigzag
      const ctrlX =
        sourceX + dx * (i / totalZigzags) + Math.cos(angle) * offsetLength
      const ctrlY =
        sourceY + dy * (i / totalZigzags) + Math.sin(angle) * offsetLength

      // Calculate the end point for this segment
      const endX = sourceX + dx * ((i + 1) / totalZigzags)
      const endY = sourceY + dy * ((i + 1) / totalZigzags)

      // Append this segment to the path data
      pathData += ` Q${ctrlX},${ctrlY} ${endX},${endY}`
    }

    return pathData
  }

  function positionLinkStraight(d: ModeConnectivityLink) {
    const x1 = xScale(d.source.x)
    const y1 = yScale(d.source.y)
    const x2 = xScale(d.target.x)
    const y2 = yScale(d.target.y)

    return "M" + x1 + "," + y1 + "L" + x2 + "," + y2
  }

  // Add a line for each link, and a circle for each node.
  const link = svgbase
    .select(".links")
    .attr("stroke", semiGlobalLocalSturctureColor.linkColor)
    .selectAll("path")
    .data(links)
    .join("path")
    .attr("class", "link")
    .attr(
      "id",
      (d) =>
        "link-" +
        d.source.modelId +
        "-" +
        d.source.modeId +
        "-" +
        d.target.modelId +
        "-" +
        d.target.modeId +
        "-link"
    )
    .attr("stroke-width", (d) => linkThickness(d.weight))
    .attr("fill", "none")
    .attr("d", (d) => {
      // return positionLinkSmooth(d)
      // return positionLinkRough(d)
      // return positionLinkStraight(d)
      return generateCurvePath(d)
      // return generateSpringPath(d)
    })

  // ploting nodes

  const innerRadius = 20
  const outerRadius = 40

  const barScale = d3
    .scaleRadial()
    .domain([
      d3.min(nodes, (d) => d3.min(d.localFlatness)),
      d3.max(nodes, (d) => d3.max(d.localFlatness)),
    ])
    .range([innerRadius, outerRadius])

  const numberOfMetrics = Object.keys(nodes[0].localMetric).length
  const barIndexScale = d3
    .scaleBand()
    .domain(d3.range(10))
    .range([0, 2 * Math.PI])

  const node = svgbase
    .selectAll(".nodes")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1)
    .selectAll(".node")
    .data(nodes)
    .join("g")
    .attr("class", "node")
    .attr("id", (d) => "nodeGroup-" + d.modelId + "-" + d.modeId)
    .attr("transform", positionNode)

  const performanceGroup = node
    .selectAll(".performanceGroup")
    .data((d) => [d])
    .join("g")
    .attr("class", "performanceGroup")

  // performanceGroup
  //   .selectAll(".performanceBackgroundRect")
  //   .data((d) => [d])
  //   .join("rect")
  //   .attr("class", "performanceBackgroundRect")
  //   .attr("width", 240)
  //   .attr("height", outerRadius * 2)
  //   .attr("fill", semiGlobalLocalSturctureColor.itemBackgroundColor)
  //   .attr("x", 0)
  //   .attr("y", -outerRadius)
  //   .attr("rx", 5)
  //   .attr("ry", 5)
  //   .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)
  //
  // performanceGroup
  //   .selectAll(".modeName")
  //   .data((d) => [d])
  //   .join("text")
  //   .attr("class", "modeName")
  //   .attr("text-anchor", "start")
  //   .attr("dominant-baseline", "central")
  //   .attr("fill", semiGlobalLocalSturctureColor.textColor)
  //   .attr("stroke", "none")
  //   .attr("font-size", 14)
  //   .text((d) => "Mode " + d.modeId.slice(-10))
  //   .attr("x", outerRadius)
  //   .attr("y", -outerRadius + 10)
  //

  // circle in the center
  performanceGroup
    .selectAll(".outerRing")
    .data((d) => [d])
    .join("circle")
    .attr("class", "outerRing")
    .attr("id", (d) => {
      return "outerRing-" + d.modelId + "-" + d.modeId
    })
    .attr("r", outerRadius + 22)
    .attr("fill", semiGlobalLocalSturctureColor.itemBackgroundColor)
    .attr("cx", 0)
    .attr("cy", 0)
    .attr("stroke", (d) => {
      if (
        selectedCheckPointIdList[modelIdIndex] ===
        d.modelId + "-" + d.modeId
      ) {
        return "red"
      }
      return semiGlobalLocalSturctureColor.strokeColor
    })
    .attr("stroke-width", (d) => {
      if (
        selectedCheckPointIdList[modelIdIndex] ===
        d.modelId + "-" + d.modeId
      ) {
        return 4
      }
      return 1
    })

  // circle in the center
  performanceGroup
    .selectAll(".middleRing")
    .data((d) => [d])
    .join("circle")
    .attr("class", "middleRing")
    .attr("id", (d) => {
      return "middleRing-" + d.modelId + "-" + d.modeId
    })
    .attr("r", outerRadius + 2)
    .attr("fill", semiGlobalLocalSturctureColor.itemInnerBackgroundColor)
    .attr("cx", 0)
    .attr("cy", 0)
    .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)

  // circle in the center
  performanceGroup
    .selectAll(".innerRing")
    .data((d) => [d])
    .join("circle")
    .attr("class", "innerRing")
    .attr("id", (d) => {
      return "innerRing-" + d.modelId + "-" + d.modeId
    })
    .attr("r", barScale(0))
    .attr("fill", semiGlobalLocalSturctureColor.itemInnerBackgroundColor)
    .attr("cx", 0)
    .attr("cy", 0)
    .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)
    .attr("stroke-width", 0.5)

  // performanceGroup
  //   .selectAll(".modeName")
  //   .data((d) => [d])
  //   .join("text")
  //   .attr("class", "modeName font-serif")
  //   .attr("text-anchor", "middle")
  //   .attr("dominant-baseline", "central")
  //   .attr("fill", semiGlobalLocalSturctureColor.textColor)
  //   .attr("stroke", "none")
  //   .attr("font-size", "1.4rem")
  //   .attr("font-weight", "bold")
  //   .text((d) => "Seed [" + d.modeId + "]")
  //   .attr("x", 0)
  //   .attr("y", 2 * outerRadius + 14)

  const performanceBarScale = d3
    .scaleLinear()
    .domain([0, 1])
    .range([0, Math.PI / 2])

  const metricArc = d3
    .arc()
    .innerRadius(outerRadius + 3)
    .outerRadius(outerRadius + 20)
    .startAngle((_d, i) => (i * (2 * Math.PI)) / numberOfMetrics)
    .endAngle(
      (d, i) =>
        (i * (2 * Math.PI)) / numberOfMetrics + performanceBarScale(d[1])
    )

  const metricArcLine = d3
    .arc()
    .innerRadius(outerRadius + 3)
    .outerRadius(outerRadius + 30)
    .startAngle((_d, i) => (i * (2 * Math.PI)) / numberOfMetrics)
    .endAngle((_d, i) => (i * (2 * Math.PI)) / numberOfMetrics)

  // Performance Bar, Acc, Prec, Recall, F1
  performanceGroup
    .selectAll(".performanceBar")
    .data((d) => Object.entries(d.localMetric))
    .join("path")
    .attr("class", "performanceBar")
    .attr("d", metricArc)
    .attr("fill", semiGlobalLocalSturctureColor.metricBarColor)
    .attr("stroke", "none")
    .on("mouseover", function (event, d) {
      d3.select(this).attr(
        "fill",
        semiGlobalLocalSturctureColor.hoverMetricBarColor
      )
      tooltip
        .style("visibility", "visible")
        .html(d[0] + ": " + roundToPercentage(d[1]))
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px")
    })
    .on("mouseout", function () {
      d3.select(this).attr("fill", semiGlobalLocalSturctureColor.metricBarColor)
      tooltip.style("visibility", "hidden")
    })

  performanceGroup
    .selectAll(".performanceBarLine")
    .data((d) => Object.entries(d.localMetric))
    .join("path")
    .attr("class", "performanceBarLine")
    .attr("d", metricArcLine)
    .attr("fill", "none")
    .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)

  const performanceTextScale = d3
    .scaleBand()
    .domain(d3.range(numberOfMetrics))
    .range([Math.PI / numberOfMetrics, 2 * Math.PI + Math.PI / numberOfMetrics])

  performanceGroup
    .selectAll(".performanceLabel")
    .data((d) => Object.entries(d.localMetric))
    .join("text")
    .attr("class", "performanceLabel font-serif")
    .attr("text-anchor", (d, i) => {
      if (numberOfMetrics % 2 === 0) {
        if (performanceTextScale(i) < Math.PI) {
          return "start"
        }
        return "end"
      } else {
        if (performanceTextScale(i) < Math.PI) {
          return "start"
        } else if (performanceTextScale(i) > Math.PI) {
          return "end"
        } else {
          return "middle"
        }
      }
    })
    .attr("fill", semiGlobalLocalSturctureColor.textColor)
    .attr("stroke", "none")
    .attr("font-size", "1.2rem")
    .attr("transform", (d, i) => {
      const angle = performanceTextScale(i) * (180 / Math.PI)
      return `rotate(${0}, ${
        Math.sin(performanceTextScale(i)) * (outerRadius + 40)
      }, ${
        -Math.cos(performanceTextScale(i)) * (outerRadius + 40)
      }) translate(0, 10)`
    })
    .attr(
      "x",
      (_d, i) => Math.sin(performanceTextScale(i)) * (outerRadius + 40)
    )
    .attr(
      "y",
      (_d, i) => -Math.cos(performanceTextScale(i)) * (outerRadius + 40)
    )
    .text((d) => d[0].slice(0, 1).toUpperCase() + d[0].slice(1))

  performanceGroup
    .selectAll(".performanceText")
    .data((d) => Object.entries(d.localMetric))
    .join("text")
    .attr("class", "performanceText")
    .attr("text-anchor", (d, i) => {
      if (numberOfMetrics % 2 === 0) {
        if (performanceTextScale(i) < Math.PI) {
          return "start"
        }
        return "end"
      } else {
        if (performanceTextScale(i) < Math.PI) {
          return "start"
        } else if (performanceTextScale(i) > Math.PI) {
          return "end"
        } else {
          return "middle"
        }
      }
    })
    .attr("fill", semiGlobalLocalSturctureColor.textColor)
    .attr("stroke", "none")
    .attr("font-size", "1rem")
    .attr("transform", (_d, i) => {
      const angle = performanceTextScale(i) * (180 / Math.PI)
      return `rotate(${0}, ${
        Math.sin(performanceTextScale(i)) * (outerRadius + 40)
      }, ${
        -Math.cos(performanceTextScale(i)) * (outerRadius + 40)
      }) translate(0, 35)`
    })
    .attr(
      "x",
      (_d, i) => Math.sin(performanceTextScale(i)) * (outerRadius + 40)
    )
    .attr(
      "y",
      (_d, i) => -Math.cos(performanceTextScale(i)) * (outerRadius + 40)
    )
    .text((d) => roundToPercentage(d[1]))

  // const performanceBarScale = d3.scaleLinear().domain([0, 1]).range([0, 90])
  // const performanceBarPositionScale = d3
  //   .scaleBand()
  //   .domain(d3.range(4))
  //   .range([-outerRadius + 20, outerRadius])
  //   .padding(0.3)
  //
  // performanceGroup
  //   .selectAll(".performanceBar")
  //   .data((d) => Object.entries(d.localMetric))
  //   .join("rect")
  //   .attr("class", "performanceBar")
  //   .attr("width", (d) => performanceBarScale(d[1]))
  //   .attr("height", (d) => performanceBarPositionScale.bandwidth())
  //   .attr("fill", semiGlobalLocalSturctureColor.metricBarColor)
  //   .attr("stroke", "none")
  //   .attr("y", (_d, i) => performanceBarPositionScale(i))
  //   .attr("x", 110)
  //
  // performanceGroup
  //   .selectAll(".performanceBarLabel")
  //   .data((d) => Object.entries(d.localMetric))
  //   .join("text")
  //   .attr("class", "performanceBarLabel")
  //   .attr("y", (_d, i) => performanceBarPositionScale(i))
  //   .attr("dy", 10)
  //   .attr("x", 105)
  //   .attr("text-anchor", "end")
  //   .attr("fill", semiGlobalLocalSturctureColor.textColor)
  //   .attr("stroke", "none")
  //   .attr("font-size", 14)
  //   .text((d) => d[0])
  //
  // performanceGroup
  //   .selectAll(".performanceBarValue")
  //   .data((d) => Object.entries(d.localMetric))
  //   .join("text")
  //   .attr("class", "performanceBarValue")
  //   .attr("y", (_d, i) => performanceBarPositionScale(i))
  //   .attr("dy", 10)
  //   .attr("x", (d) => 115 + performanceBarScale(d[1]))
  //   .attr("text-anchor", "start")
  //   .attr("fill", semiGlobalLocalSturctureColor.textColor)
  //   .attr("stroke", "none")
  //   .attr("font-size", 14)
  //   .text((d) => roundToPercentage(d[1]))

  const arc = d3
    .arc()
    .innerRadius((d) => {
      return barScale(Math.min(Number(d), 0))
    })
    .outerRadius((d) => barScale(Math.max(Number(d), 0)))
    .startAngle((_d, i) => barIndexScale(i))
    .endAngle((_d, i) => barIndexScale(i) + barIndexScale.bandwidth())
    .padAngle(1.5 / innerRadius)
    .padRadius(innerRadius)

  node
    .selectAll(".bar")
    .data((d) => d.localFlatness)
    .join("path")
    .attr("class", "bar")
    .attr("data-index", (_d, i) => i + 1)
    .attr("fill", (d) => {
      if (d > 0) {
        return semiGlobalLocalSturctureColor.radioBarColor
      } else {
        return "#cdcdcd"
      }
    })
    .attr("d", arc)
    .attr("stroke", "none")
    .on("mouseover", function (event, d) {
      d3.select(this).attr(
        "fill",
        semiGlobalLocalSturctureColor.hoverRadioBarColor
      )
      const i = d3.select(this).attr("data-index")
      tooltip
        .style("visibility", "visible")
        .html("# " + i + " Hessian Eigenvalue: " + d)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px")
    })
    .on("mouseout", function () {
      d3.select(this).attr("fill", (d) => {
        if (d > 0) {
          return semiGlobalLocalSturctureColor.radioBarColor
        } else {
          return "#cdcdcd"
        }
      })
      tooltip.style("visibility", "hidden")
    })

  node
    .selectAll(".circle")
    .data((d) => [d])
    .join("circle")
    .attr("class", "circle")
    .attr("id", (d) => "circle-" + d.modelId + "-" + d.modeId)
    .attr("r", 13)
    .attr("fill", (d) => modelColorMap[d.modelId])
    .attr("stroke", "none")
    .attr("cx", 0)
    .attr("cy", 0)
    .on("mouseover", (_event, d) => {
      svgbase.selectAll(".node").style("opacity", 0.1)
      svgbase
        .select("#nodeGroup-" + d.modelId + "-" + d.modeId)
        .style("opacity", 1)
        .raise()

      // svgbase.selectAll(".circle").style("opacity", 0.2)
      // svgbase.selectAll(".bar").style("opacity", 0.2)
      // svgbase.selectAll(".outerRing").style("opacity", 0.2)
      // svgbase.selectAll(".middleRing").style("opacity", 0.2)
      // svgbase.selectAll(".innerRing").style("opacity", 0.2)
      // svgbase.selectAll(".performanceBar").style("opacity", 0.2)
      // svgbase.selectAll(".performanceBarLine").style("opacity", 0.2)
      // svgbase.selectAll(".performanceText").attr("visibility", "hidden")
      // svgbase.selectAll(".performanceLabel").attr("visibility", "hidden")
      svgbase.selectAll(".link").style("stroke-opacity", 0)
      const sourceSelector = `[id^='link-${d.modelId + "-" + d.modeId}-']` // IDs that start with link-yourSpecificId-
      const targetSelector = `[id$='-${d.modelId + "-" + d.modeId}-link']` // IDs that end with -yourSpecificId

      svgbase
        .select("#circle-" + d.modelId + "-" + d.modeId)
        .style("cursor", "pointer")
        .style("opacity", 1)

      svgbase
        // .selectAll(".link")
        .selectAll(`${sourceSelector}, ${targetSelector}`)
        .style("stroke-opacity", 1)
        .raise()
      // svgbase
      //   .select("#outerRing-" + d.modelId + "-" + d.modeId)
      //   .style("opacity", 1)
      // svgbase
      //   .select("#middleRing-" + d.modelId + "-" + d.modeId)
      //   .style("opacity", 1)
      // svgbase
      //   .select("#innerRing-" + d.modelId + "-" + d.modeId)
      //   .style("opacity", 1)
    })
    .on("mouseout", (_event, d) => {
      svgbase.selectAll(".node").style("opacity", 1)
      svgbase.selectAll(".link").style("stroke-opacity", 1)
      // svgbase.selectAll(".circle").style("opacity", 1)
      // svgbase.selectAll(".bar").style("opacity", 1)
      // svgbase.selectAll(".outerRing").style("opacity", 1)
      // svgbase.selectAll(".middleRing").style("opacity", 1)
      // svgbase.selectAll(".innerRing").style("opacity", 1)
      // svgbase.selectAll(".performanceBar").style("opacity", 1)
      // svgbase.selectAll(".performanceBarLine").style("opacity", 1)
      // svgbase.selectAll(".performanceText").attr("visibility", "visible")
      // svgbase.selectAll(".performanceLabel").attr("visibility", "visible")
      svgbase
        .select("#" + d.modelId + "-" + d.modeId)
        .style("cursor", "default")
    })
    .on("click", (_event, d) => {
      svgbase.selectAll(".node").style("opacity", 1)
      svgbase.selectAll(".link").style("stroke-opacity", 1)
      // svgbase.selectAll(".node").style("opacity", 0.1)
      // svgbase
      //   .select("#nodeGroup-" + d.modelId + "-" + d.modeId)
      //   .style("opacity", 1)
      //   .raise()
      // svgbase.selectAll(".link").style("stroke-opacity", 0)
      // const sourceSelector = `[id^='link-${d.modelId + "-" + d.modeId}-']` // IDs that start with link-yourSpecificId-
      // const targetSelector = `[id$='-${d.modelId + "-" + d.modeId}-link']` // IDs that end with -yourSpecificId
      //
      // svgbase
      //   .select("#circle-" + d.modelId + "-" + d.modeId)
      //   .style("cursor", "pointer")
      //   .style("opacity", 1)
      //
      // svgbase
      //   // .selectAll(".link")
      //   .selectAll(`${sourceSelector}, ${targetSelector}`)
      //   .style("stroke-opacity", 1)
      //   .raise()
      svgbase
        .selectAll("[id^=outerRing-" + d.modelId + "]")
        .attr("stroke-width", 1)
        .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)
      updateSelectedModelIdModeId(modelIdIndex, d.modelId + "-" + d.modeId)
    })

  // Zooming

  const zoom = d3
    .zoom()
    .on("zoom", zoomed)
    .extent([
      [0, 0],
      [width, height],
    ])
    .scaleExtent([0.5, 8])

  svgbase.call(zoom)

  const zoomContainer = svgbase.select(".zoom-container")

  function zoomed(event) {
    zoomContainer.attr("transform", event.transform)
  }

  svgbase
    .call(zoom.transform, d3.zoomIdentity.scale(1))
    .transition()
    .duration(0)
    .call(
      zoom.transform,
      d3.zoomIdentity.scale(0.7).translate(width / 6, height / 6)
    )
}

export default function SemiGlobalLocalCore({
  width,
  height,
  data,
  selectedCheckPointIdList,
  updateSelectedModelIdModeId,
  modelId,
  modelIdIndex,
}: SemiGlobalLocalCoreProp): React.JSX.Element {
  // NOTE: modelUIList is no longer useful
  const svg = React.useRef<SVGSVGElement>(null)
  const wraperRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    // Function to update the D3 chart based on the div dimensions
    const updateChart = () => {
      // Get the div dimensions using the ref
      const divElement = wraperRef.current
      const width = divElement.clientWidth
      const height = divElement.clientHeight
      const svgE = d3.select(svg.current)
      svgE.attr("width", width).attr("height", height)
    }

    // Call the updateChart function initially and on window resize
    updateChart()
    window.addEventListener("resize", updateChart)

    return () => {
      window.removeEventListener("resize", updateChart)
    }
  }, [])

  React.useEffect(() => {
    if (!data) return
    const clonedData = JSON.parse(JSON.stringify(data))
    render(
      svg,
      wraperRef,
      clonedData,
      selectedCheckPointIdList,
      updateSelectedModelIdModeId,
      modelId,
      modelIdIndex
    )
  }, [
    data,
    width,
    height,
    selectedCheckPointIdList,
    updateSelectedModelIdModeId,
    modelId,
    modelIdIndex,
  ])

  return (
    <div ref={wraperRef} className="h-full w-full rounded border">
      <svg ref={svg}>
        <g className="zoom-container">
          <g className="links"></g>
          <g className="nodes"></g>
        </g>
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
