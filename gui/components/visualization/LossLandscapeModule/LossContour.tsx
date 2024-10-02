"use client"

import * as React from "react"
import * as d3 from "d3"

import { GlobalInfo, LossLandscape } from "@/types/losslens"
import { lossContourColor } from "@/styles/vis-color-scheme"

interface LossContourCoreProp {
    data: LossLandscape
    dimensions: { width: number; height: number }
    globalInfo: GlobalInfo
}

function calculatePadding(width: number, height: number) {
    return {
        top: height * 0.02,
        right: width * 0.05,
        bottom: height * 0.01,
        left: width * 0.05,
    }
}

function render(
    svgRef: React.RefObject<SVGSVGElement>,
    wrapperRef: React.RefObject<HTMLDivElement>,
    data: LossLandscape,
    globalInfo: GlobalInfo,
    dimensions: { width: number; height: number }
) {
    const { width, height } = dimensions

    const padding = calculatePadding(width, height)
    const h = height - padding.top - padding.bottom
    const w = width - padding.left - padding.right

    const globalUpperBound = globalInfo.lossBounds.upperBound
    const globalLowerBound = globalInfo.lossBounds.lowerBound
    const upperBound = data.grid.flat().reduce((a, b) => Math.max(a, b))
    const lowerBound = data.grid.flat().reduce((a, b) => Math.min(a, b))

    const svg = d3
        .select(svgRef.current)
        .attr("width", width)
        .attr("height", height)
        .select("g")
        .attr("transform", `translate(${padding.left},${padding.top})`)

    const lengthOfGrid = data.grid.length
    const q = w / lengthOfGrid

    const thresholdArray = []
    for (let i = 0; i < 30; i++) {
        const threshold = lowerBound + (i / 29) * (upperBound - lowerBound)
        thresholdArray.push(threshold)
    }

    const customizedColors = lossContourColor.gridColor

    const domainSteps = customizedColors.map((_color, i) => {
        return (
            globalUpperBound -
            (i * (globalUpperBound - globalLowerBound)) /
                customizedColors.length
        )
    })

    const color = d3
        .scaleLinear()
        .domain(domainSteps)
        .range(customizedColors.reverse())

    const gridX = -q
    const gridY = -q
    const gridK = q

    const transform = ({ type, value, coordinates }) => {
        return {
            type,
            value,
            coordinates: coordinates.map((rings) => {
                return rings.map((points) => {
                    return points.map(([x, y]) => [
                        gridX + gridK * x,
                        gridY + gridK * y,
                    ])
                })
            }),
        }
    }

    const contours = d3
        .contours()
        .size([40, 40])
        .thresholds(thresholdArray)(data.grid.flat())
        .map(transform)

    svg.selectAll("path")
        .data(contours)
        .join("path")
        .attr("fill", (d) => color(d.value))
        .attr("stroke", "#333")
        .attr("stroke-opacity", 0.5)
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
        .attr("y", h - 7)
        .attr("x", 35)
        .attr("height", 7)
        .attr("width", w - 45)
        .attr("stroke", "#666")
        .attr("fill", "url(#lossgrad)")

    const legendScale = d3
        .scaleLinear()
        .range([0, w - 45])
        .domain([globalLowerBound, globalUpperBound])

    const legendAxis = svg.selectAll(".legendAxis").data([data])

    legendAxis
        .join("g")
        .attr("class", "legendAxis")
        .attr("transform", `translate(35, ${h})`)
        .call(d3.axisTop(legendScale).ticks(4).tickFormat(d3.format(".2s")))

    legendAxis
        .selectAll(".tick text")
        .attr("font-size", "0.7rem")
        .attr("fill", "#000")
        .attr("text-anchor", "middle")
    legendAxis.selectAll(".tick line").attr("stroke", "#000")

    legendAxis
        .selectAll(".legendLabel")
        .data([1])
        .join("text")
        .attr("class", "legendLabel")
        .attr("x", -20)
        .attr("y", 0)
        .attr("font-size", "0.8rem")
        .attr("font-weight", "semibold")
        .attr("fill", "#000")
        .text("Loss")
}

export default function LossContourCore({
    dimensions,
    data,
    globalInfo,
}: LossContourCoreProp): React.JSX.Element {
    const svg = React.useRef<SVGSVGElement>(null)
    const wrapperRef = React.useRef<HTMLDivElement>(null)

    React.useEffect(() => {
        if (wrapperRef.current && svg.current && data) {
            const wrapperWidth = wrapperRef.current.clientWidth
            const wrapperHeight = wrapperRef.current.clientHeight
            const size = Math.min(wrapperWidth, wrapperHeight)

            render(svg, wrapperRef, data, globalInfo, {
                width: size,
                height: size,
            })
        }
    }, [data, dimensions, globalInfo])

    return (
        <div
            ref={wrapperRef}
            className="flex h-full w-full flex-col items-center justify-start"
        >
            <div className="text-center text-sm">
                Local Structure - Loss Contour
            </div>
            <svg ref={svg}>
                <g></g>
            </svg>
        </div>
    )
}
