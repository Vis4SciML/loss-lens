"use client"

import * as React from "react"
import * as d3 from "d3"
import { useAtom } from "jotai"

import { Graph, Link, Node } from "@/types/rsdn"
import { activatedTabAtom, getActivatedTabAtom } from "@/lib/store"

interface CustomDragEvent extends DragEvent {
  active: boolean
  subject: {
    fx: number | null
    fy: number | null
    x: number | null
    y: number | null
  }
}

interface SupplyChainCoreProps {
  data: Graph
  height: number
  width: number
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  data: Graph,
  width: number,
  height: number
) {
  // const width = 1600
  // const height = 850
  const svgbase = d3.select(svgRef.current)
  svgbase.attr("width", width).attr("height", height)
  console.log("data in render: ")
  console.log(data)

  const links = data.links
  const nodes = data.nodes

  function dragstarted(event: CustomDragEvent) {
    if (!event.active) simulation.alphaTarget(0.6).restart()
    event.subject.fx = event.subject.x
    event.subject.fy = event.subject.y
  }

  function dragged(event: CustomDragEvent) {
    event.subject.fx = event.x
    event.subject.fy = event.y
  }

  function dragended(event: CustomDragEvent) {
    if (!event.active) simulation.alphaTarget(0)
    event.subject.fx = null
    event.subject.fy = null
  }

  const ticked = () => {
    link.attr("d", positionLink)
    node.attr("cx", (d: Node) => d.x).attr("cy", (d: Node) => d.y)
    label.attr("x", (d: Node) => d.x).attr("y", (d: Node) => d.y + 20)
  }

  function positionLink(d: Link) {
    let offset = 30

    let midpoint_x = (d.source.x! + d.target.x!) / 2
    let midpoint_y = (d.source.y! + d.target.y!) / 2

    let dx = d.target.x! - d.source.x!
    let dy = d.target.y! - d.source.y!

    let normalise = Math.sqrt(dx * dx + dy * dy)

    let offSetX = midpoint_x + offset * (dy / normalise)
    let offSetY = midpoint_y - offset * (dx / normalise)

    return (
      "M" +
      d.source.x +
      "," +
      d.source.y +
      "S" +
      offSetX +
      "," +
      offSetY +
      " " +
      d.target.x +
      "," +
      d.target.y
    )
  }

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d: Link) => d.id)
        .distance(100)
        .strength(2)
    )
    .force("charge", d3.forceManyBody().strength(-800))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .on("tick", ticked)

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
    .attr("fill", "#999")
    .attr("opacity", 1)
    .style("stroke", "none")

  // Add a line for each link, and a circle for each node.
  const link = svgbase
    .select(".links")
    .attr("stroke", "#ccc")
    .attr("stroke-opacity", 0.6)
    .selectAll("path")
    .data(links)
    .join("path")
    .attr("stroke-width", () => 2)
    .attr("fill", "none")
    .attr("marker-end", "url(#arrowhead)")

  const label = svgbase
    .selectAll(".labels")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .attr("fill", "#333")
    .attr("font-size", 12)
    .text((d) => d.id.slice(-10))

  const node = svgbase
    .selectAll(".nodes")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", 7)
    .attr("fill", () => "#cacaca")

  node.call(
    d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended)
  )

  label.call(
    d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended)
  )

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

export default function SupplyChainCore({
  width,
  height,
  data,
}: SupplyChainCoreProps): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)
  const [activatedTab] = useAtom(getActivatedTabAtom)

  console.log("activated tab: " + activatedTab)

  React.useEffect(() => {
    if (!data) return
    console.log("rendering")
    console.log(data)
    const clonedData = JSON.parse(JSON.stringify(data))
    render(svg, clonedData, width - 450, height - 130)
  }, [data, width, height, activatedTab])

  return (
    <svg ref={svg}>
      <g className="zoom-container">
        <g className="links"></g>
        <g className="nodes"></g>
        <g className="labels"></g>
      </g>
    </svg>
  )
}
