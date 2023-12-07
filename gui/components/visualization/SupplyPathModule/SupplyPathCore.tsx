"use client"

import * as React from "react"
import * as d3 from "d3"
import { useAtom } from "jotai"

import { NodeStructure, PathStructure, SupplyPath } from "@/types/rsdn"
import { getActivatedTabAtom } from "@/lib/store"

interface SupplyPathCoreProps {
  data: SupplyPath
  height: number
  width: number
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  data: SupplyPath,
  width: number,
  height: number
) {
  // const width = 1600
  // const height = 850
  const svgbase = d3.select(svgRef.current)
  svgbase.attr("width", width).attr("height", height)

  const nodeStructure = data.nodeStructure
  const edgeStructure = data.pathStructure

  console.log("nodeStructure in render: ")
  console.log(nodeStructure)

  const nodeLayers = svgbase
    .select(".nodes")
    .selectAll(".layers")
    .data(nodeStructure)
    .join("g")
    .attr("class", "layers")
    .attr("transform", (_d, i) => `translate(${i * 200 + 50}, 0)`)

  nodeLayers
    .selectAll("circle")
    .data((d: NodeStructure[]) => d)
    .join("circle")
    .attr("r", 7)
    .attr("fill", () => "#cacaca")
    .attr("stroke", "#fff")
    .attr("cx", 0)
    .attr("cy", (d: NodeStructure) => d.cy)

  nodeLayers
    .selectAll("text")
    .data((d: NodeStructure[]) => d)
    .join("text")
    .attr("x", 0)
    .attr("y", (d: NodeStructure) => d.cy + 20)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("fill", "#333")
    .text((d: NodeStructure) => d.id.slice(-10))

  const edgeLayers = svgbase
    .select(".edges")
    .selectAll(".layers")
    .data(edgeStructure)
    .join("g")
    .attr("class", "layers")
    .attr("transform", (_d, i) => `translate(${i} + 50, 0)`)

  const curve = d3.line().curve(d3.curveBasis)

  edgeLayers
    .selectAll("path")
    .data((d: PathStructure[]) => d)
    .join("path")
    .attr("stroke-width", 1)
    .attr("stroke", "#ccc")
    .attr("fill", "none")
    .attr("d", (d: PathStructure) => {
      if (d.y1 === d.y2) {
        return curve([
          [d.x1, d.y1],
          [d.x2, d.y2],
        ])
      }
      return curve([
        [d.x1, d.y1],
        [d.x1 + (d.x2 - d.x1) / 2, d.y1],
        [d.x1 + (d.x2 - d.x1) / 2, d.y2],
        [d.x2, d.y2],
      ])
    })
    .attr("marker-end", "url(#arrowhead)")

  svgbase
    .append("defs")
    .append("marker")
    .attr("id", "arrowhead")
    .attr("markerUnits", "userSpaceOnUse")
    .attr("viewBox", "-0 -5 10 10")
    .attr("refX", 20)
    .attr("refY", 0)
    .attr("orient", "auto")
    .attr("markerWidth", 8)
    .attr("markerHeight", 8)
    .attr("xoverflow", "visible")
    .append("svg:path")
    .attr("d", "M 0,-5 L 10 ,0 L 0,5")
    .attr("fill", "#cacaca")
    .attr("opacity", 1)
    .style("stroke", "none")

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
}

export default function SupplyPathCore({
  data,
  width,
  height,
}: SupplyPathCoreProps): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)

  const [activatedTab] = useAtom(getActivatedTabAtom)

  React.useEffect(() => {
    if (!data) return
    render(svg, data, width * 0.82, height - 130)
  }, [data, width, height, activatedTab])

  return (
    <svg ref={svg}>
      <g className="zoom-container">
        <g className="edges"></g>
        <g className="nodes"></g>
      </g>
    </svg>
  )
}
