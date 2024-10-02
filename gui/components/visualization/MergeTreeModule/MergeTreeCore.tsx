"use client"

import * as React from "react"
import * as d3 from "d3"

import { MergeTreeData } from "@/types/losslens"
import { mergeTreeColor } from "@/styles/vis-color-scheme"

interface MergeTreeCoreProp {
    data: MergeTreeData
    dimensions: { width: number; height: number }
}

function calculatePadding(width: number, height: number) {
    return {
        top: height * 0.02,
        right: width * 0.3,
        bottom: height * 0.1,
        left: width * 0.25,
    }
}

function render(
    svgRef: React.RefObject<SVGSVGElement>,
    wrapperRef: React.RefObject<HTMLDivElement>,
    data: MergeTreeData,
    dimensions: { width: number; height: number }
) {
    const { width, height } = dimensions

    const padding = calculatePadding(width, height)
    const h = height - padding.top - padding.bottom
    const w = width - padding.left - padding.right

    const svg = d3
        .select(svgRef.current)
        .attr("width", width)
        .attr("height", height)
        .select("g")
        .attr("transform", `translate(${padding.left},${padding.top})`)

    /**
     * Scales, dynamically based on data.
     */

    const nodes = data.nodes
    const edges = data.edges

    const xMin = d3.min(edges, (d) => Math.min(d.sourceX, d.targetX))
    const xMax = d3.max(edges, (d) => Math.max(d.sourceX, d.targetX))
    const yMin = d3.min(edges, (d) => Math.min(d.sourceY, d.targetY))
    const yMax = d3.max(edges, (d) => Math.max(d.sourceY, d.targetY))

    const xScale = d3
        .scaleLinear()
        .range([0, w])
        .domain(d3.extent([xMin, xMax]))

    let zoom = d3.zoom().on("zoom", handleZoom)

    function handleZoom(e) {
        svg.attr("transform", e.transform)
    }

    d3.select(svgRef.current).call(zoom)

    const yScale = d3
        .scaleLinear()
        .range([h, 0])
        .domain(d3.extent([yMin, yMax]))

    svg.selectAll(".merge-tree-edge")
        .data(edges)
        .join("line")
        .attr("class", "merge-tree-edge")
        .attr("x1", (d) => xScale(d.sourceX))
        .attr("y1", (d) => yScale(d.sourceY))
        .attr("x2", (d) => xScale(d.targetX))
        .attr("y2", (d) => yScale(d.targetY))
        .attr("stroke", mergeTreeColor.strokeColor)
        .attr("stroke-width", 1)
    // .attr('opacity', 0.5)

    // svg
    //   .selectAll('circle')
    //   .data(nodes)
    //   .join('circle')
    //   .attr('cx', (d) => xScale(d.x))
    //   .attr('cy', (d) => yScale(d.y))
    //   .attr('r', 3)
    //   .attr('fill', '#666')
    //   .attr('stroke', 'black')
    //   .attr('stroke-width', 1)
    //   .attr('opacity', 1)
    //

    // Add y-axis on the right side with sparse, scientific ticks
    const yAxis = d3.axisRight(yScale).ticks(5).tickFormat(d3.format(".2e"))

    if (svg.select(".y-axis").empty()) {
        svg.append("g")
            .attr("class", "y-axis")
            .attr("transform", `translate(${w + 15}, 0)`)
            .call(yAxis)
            .append("text")
            .attr("class", "y-axis-label")
            .attr("fill", "black")
            .attr("text-anchor", "end")
            .attr("x", 30)
            .attr("y", h)
            .text("Loss")
    }
}

export default function MergeTreeCore({
    dimensions,
    data,
}: MergeTreeCoreProp): React.JSX.Element {
    const svg = React.useRef<SVGSVGElement>(null)
    const wrapperRef = React.useRef<HTMLDivElement>(null)

    React.useEffect(() => {
        if (wrapperRef.current && svg.current && data) {
            const wrapperWidth = wrapperRef.current.clientWidth
            const wrapperHeight = wrapperRef.current.clientHeight
            const size = Math.min(wrapperWidth, wrapperHeight)

            render(svg, wrapperRef, data, { width: size, height: size })
        }
    }, [data, dimensions])

    return (
        <div
            ref={wrapperRef}
            className="flex h-full w-full flex-col items-center justify-start"
        >
            <div className="text-center text-sm">Merge Tree</div>
            <svg ref={svg}>
                <g></g>
            </svg>
        </div>
    )
}
