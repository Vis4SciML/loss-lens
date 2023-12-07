"use client"

import * as React from "react"
import * as d3 from "d3"

export interface ModelEnsemblePoint {
  x: number
  y: number
}

export interface ModelEnsembleDataType {
  data: ModelEnsemblePoint[] | null
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  data: ModelEnsemblePoint[] | null
) {
  const svgbase = d3.select(svgRef.current)

  const circleGroup = svgbase.select(".circle-group")

  circleGroup
    .selectAll("circle")
    .data(data as ModelEnsemblePoint[])
    .join("circle")
    .attr("r", 3)
    .attr("fill", "steelblue")
    .attr("cx", (d: any) => d.x)
    .attr("cy", (d: any) => d.y)
}

export default function ModelEnsemble({
  data,
}: ModelEnsembleDataType): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)

  React.useEffect(() => {
    if (!data) return
    render(svg, data)
  }, [data])

  return (
    <svg ref={svg} viewBox="0 0 300 300">
      <g className="circle-group" />
    </svg>
  )
}
