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

function calculatePadding(width: number, height: number) {
    return {
        top: height * 0.02,
        right: width * 0.1, // Increased right margin
        bottom: height * 0.22,
        left: width * 0.05,
    }
}

function render(
    svgRef: React.RefObject<SVGSVGElement>,
    wrapperRef: React.RefObject<HTMLDivElement>,
    data: PersistenceBarcode,
    height: number,
    width: number
) {
    const padding = calculatePadding(width, height)
    const h = height - padding.top - padding.bottom
    const w = width - padding.left - padding.right

    const svg = d3
        .select(svgRef.current)
        .attr("width", width)
        .attr("height", height)
        .select("g")
        .attr("transform", `translate(${padding.left},${padding.top})`)

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

    function handleZoom(e: any) {
        svg.attr("transform", e.transform)
    }

    d3.select(svgRef.current).call(zoom as any)

    const mi = d3.min(edges.map((e) => e.y0)) as number
    const ma = d3.max(edges.map((e) => e.y1)) as number
    const xScale = d3.scaleLinear().range([0, w]).domain([mi, ma])

    const yScale = d3.scaleLinear().domain([mi, ma]).range([h, 0])

    svg.selectAll(".persistenceline")
        .data(edges)
        .join("line")
        .attr("class", "persistenceline")
        .attr("x1", (d) => xScale(d.x))
        .attr("y1", (d) => yScale(d.y0))
        .attr("x2", (d) => xScale(d.x))
        .attr("y2", (d) => yScale(d.y1))
        .attr("stroke", persistenceBarcodeColor.strokeColor)
        .attr("stroke-width", 1)

    svg.selectAll(".diag")
        .data([1])
        .join("line")
        .attr("class", "diag")
        .attr("x1", () => xScale(mi))
        .attr("y1", () => yScale(mi))
        .attr("x2", () => xScale(ma))
        .attr("y2", () => yScale(ma))
        .attr("stroke", persistenceBarcodeColor.strokeColor)
        .attr("stroke-width", 1)

    const xGroupBase1 = svg.selectAll(".x-axis1").data([1])

    const xGroup1 = xGroupBase1.join("g").attr("class", "x-axis1")

    xGroup1
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format(".2e")))
        .attr("transform", `translate(0, ${h})`)

    xGroup1
        .selectAll(".tick text")
        .attr("transform", "rotate(-25)")
        .style("text-anchor", "middle")
        .attr("dy", "0.6em")
        .attr("dx", "-0.45em")

    const xGroupBase2 = svg.selectAll(".xlabel").data([1])

    xGroupBase2
        .join("text")
        .text("Death")
        .attr("class", "xlabel")
        .attr("text-anchor", "middle")
        .attr("transform", `rotate(90, ${w - 10}, ${h / 2 - 10})`)
        .attr("x", w + 5)
        .attr("y", h / 2 - 5)
        .style("font-size", "0.8rem")
        .style("fill", persistenceBarcodeColor.textColor)

    const yGroupBase1 = svg.selectAll(".y-axis1").data([1])

    const yGroup1 = yGroupBase1.join("g").attr("class", "y-axis1")

    yGroup1
        .call(d3.axisRight(yScale).ticks(5).tickFormat(d3.format(".2e")))
        .attr("transform", `translate(${w},0)`)

    yGroup1
        .selectAll(".tick text")
        .attr("transform", "rotate(67)")
        .style("text-anchor", "middle")
        .attr("dy", "-0.9em")
        .attr("dx", "-0.2em")

    const yGroupBase2 = svg.selectAll(".ylabel").data([1])

    yGroupBase2
        .join("text")
        .attr("class", "ylabel")
        .text("Birth")
        .attr("x", w / 2)
        .attr("y", h - 10)
        .attr("text-anchor", "middle")
        .style("font-size", "0.8rem")
        .style("fill", "#000")

    xGroup1.selectAll("path").attr("stroke", "#000")
    xGroup1.selectAll(".tick line").attr("display", "block")
    xGroup1.selectAll(".tick text").attr("display", "block")
    yGroup1.selectAll("path").attr("stroke", "#000")
    yGroup1.selectAll(".tick line").attr("display", "block")
    yGroup1.selectAll(".tick text").attr("display", "block")
}

export default function PersistenceBarcodeCore({
    height,
    width,
    data,
}: PersistenceBarcodeCoreProp): React.JSX.Element {
    const svg = React.useRef<SVGSVGElement>(null)
    const wrapperRef = React.useRef<HTMLDivElement>(null)

    React.useEffect(() => {
        if (wrapperRef.current && svg.current && data) {
            render(svg, wrapperRef, data, height, width)
        }
    }, [data, height, width])

    return (
        <div
            ref={wrapperRef}
            className=" flex h-full w-full flex-col items-center justify-center"
        >
            <div className="text-center text-sm">Persistence Diagram</div>
            <svg ref={svg}>
                <g></g>
            </svg>
        </div>
    )
}
