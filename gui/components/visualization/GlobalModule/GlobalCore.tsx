"use client"

import * as React from "react"
import * as d3 from "d3"

import { ModeNode, SemiGlobalLocalStructure } from "@/types/losslens"
import { roundToPercentage } from "@/lib/utils"
import {
    modelColor,
    semiGlobalLocalSturctureColor,
} from "@/styles/vis-color-scheme"

interface GlobalCoreProp {
    data: SemiGlobalLocalStructure
    height: number
    width: number
    selectedCheckPointIdList: string[]
    updateSelectedModelIdModeId: (index: number, id: string) => void
    modelId: string
    modelIdIndex: number
    modelMetaData: any
}

function render(
    svgRef: React.RefObject<SVGSVGElement>,
    wraperRef: React.RefObject<HTMLDivElement>,
    data: SemiGlobalLocalStructure,
    selectedCheckPointIdList: string[],
    updateSelectedModelIdModeId: (index: number, id: string) => void,
    modelId: string,
    modelIdIndex: number,
    modelMetaData: any
) {
    const divElement = wraperRef.current
    const width = divElement?.clientWidth || 0
    const height = divElement?.clientHeight || 0
    const svgbase = d3.select(svgRef.current)
    const margin = { top: 10, right: 10, bottom: 10, left: 10 }
    const h = height - margin.top - margin.bottom
    const w = width - margin.left - margin.right

    svgbase.attr("width", width).attr("height", height)
    const { nodes: wholeNodes, links: wholeLinks, modelList } = data

    const xScale = d3
        .scaleLinear()
        .domain(d3.extent(wholeNodes, (node) => node.x) as [number, number])
        .range([0, w])

    const yScale = d3
        .scaleLinear()
        .domain(d3.extent(wholeNodes, (node) => node.y) as [number, number])
        .range([0, h])

    const linkThickness = d3
        .scaleLinear()
        .domain(
            d3.extent(wholeLinks, (link) => link.weight) as [number, number]
        )
        .range([2, 20])

    const linkSmoothness = d3
        .scaleLinear()
        .domain(
            d3.extent(wholeLinks, (link) => link.weight) as [number, number]
        )
        .range([1.0, 0.1])

    const modelColorMap = modelList.reduce(
        (acc: { [key: string]: string }, model, index) => {
            acc[model] = modelColor[index]
            return acc
        },
        {}
    )

    const tooltip = d3
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
        (link) =>
            link.source.modelId === modelId && link.target.modelId === modelId
    )

    const nodes = wholeNodes.filter((node) => node.modelId === modelId)

    const positionNode = (d: ModeNode) =>
        `translate(${xScale(d.x)},${yScale(d.y)})`

    const generateCurvePath = (edge) => {
        const [scaledSourceX, scaledSourceY] = [
            xScale(edge.source.x),
            yScale(edge.source.y),
        ]
        const [scaledTargetX, scaledTargetY] = [
            xScale(edge.target.x),
            yScale(edge.target.y),
        ]
        const [midX, midY] = [
            (scaledSourceX + scaledTargetX) / 2,
            (scaledSourceY + scaledTargetY) / 2,
        ]
        const [dx, dy] = [
            scaledTargetX - scaledSourceX,
            scaledTargetY - scaledSourceY,
        ]
        const offsetLength =
            Math.sqrt(dx * dx + dy * dy) * linkSmoothness(edge.weight)
        const [offsetX, offsetY] = [
            -dy * (offsetLength / Math.sqrt(dx * dx + dy * dy)),
            dx * (offsetLength / Math.sqrt(dx * dx + dy * dy)),
        ]
        const [ctrlX, ctrlY] = [midX + offsetX, midY + offsetY]
        return `M${scaledSourceX},${scaledSourceY} Q${ctrlX},${ctrlY} ${scaledTargetX},${scaledTargetY}`
    }

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
                `link-${d.source.modelId}-${d.source.modeId}-${d.target.modelId}-${d.target.modeId}-link`
        )
        .attr("stroke-width", (d) => linkThickness(d.weight))
        .attr("fill", "none")
        .attr("d", generateCurvePath)
        .on("mouseover", function (event, d) {
            d3.select(this).attr(
                "stroke",
                semiGlobalLocalSturctureColor.hoverLinkColor
            )
            tooltip
                .style("visibility", "visible")
                .html(
                    `Source: ${d.source.modeId}<br>Target: ${d.target.modeId}<br>Mode Connectivity: ${d.weight}`
                )
                .style("top", `${event.pageY - 10}px`)
                .style("left", `${event.pageX + 10}px`)
        })
        .on("mouseout", function () {
            d3.select(this).attr(
                "stroke",
                semiGlobalLocalSturctureColor.linkColor
            )
            tooltip.style("visibility", "hidden")
        })

    const innerRadius = 20
    const outerRadius = 40

    const barScale = d3
        .scaleRadial()
        .domain([
            d3.min(nodes, (d) => d3.min(d.localFlatness) ?? 0) as number,
            d3.max(nodes, (d) => d3.max(d.localFlatness) ?? 0) as number,
        ])
        .range([innerRadius, outerRadius])

    const numberOfMetrics = Object.keys(nodes[0].localMetric).length
    const barIndexScale = d3
        .scaleBand()
        .domain(d3.range(10).map(String))
        .range([0, 2 * Math.PI])

    const node = svgbase
        .selectAll(".nodes")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1)
        .selectAll(".node")
        .data(nodes)
        .join("g")
        .attr("class", "node")
        .attr("id", (d) => `nodeGroup-${d.modelId}-${d.modeId}`)
        .attr("transform", positionNode)

    const performanceGroup = node
        .selectAll(".performanceGroup")
        .data((d) => [d])
        .join("g")
        .attr("class", "performanceGroup")

    performanceGroup
        .selectAll(".outerRing")
        .data((d) => [d])
        .join("circle")
        .attr("class", "outerRing")
        .attr("id", (d) => `outerRing-${d.modelId}-${d.modeId}`)
        .attr("r", outerRadius + 22)
        .attr("fill", semiGlobalLocalSturctureColor.itemBackgroundColor)
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("stroke", (d) =>
            selectedCheckPointIdList[modelIdIndex] ===
            `${d.modelId}-${d.modeId}`
                ? "red"
                : semiGlobalLocalSturctureColor.strokeColor
        )
        .attr("stroke-width", (d) =>
            selectedCheckPointIdList[modelIdIndex] ===
            `${d.modelId}-${d.modeId}`
                ? 4
                : 1
        )

    performanceGroup
        .selectAll(".middleRing")
        .data((d) => [d])
        .join("circle")
        .attr("class", "middleRing")
        .attr("id", (d) => `middleRing-${d.modelId}-${d.modeId}`)
        .attr("r", outerRadius + 2)
        .attr("fill", semiGlobalLocalSturctureColor.itemInnerBackgroundColor)
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)

    performanceGroup
        .selectAll(".innerRing")
        .data((d) => [d])
        .join("circle")
        .attr("class", "innerRing")
        .attr("id", (d) => `innerRing-${d.modelId}-${d.modeId}`)
        .attr("r", barScale(0))
        .attr("fill", semiGlobalLocalSturctureColor.itemInnerBackgroundColor)
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)
        .attr("stroke-width", 0.5)

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
                (i * (2 * Math.PI)) / numberOfMetrics +
                performanceBarScale(d[1])
        )

    const metricArcLine = d3
        .arc()
        .innerRadius(outerRadius + 3)
        .outerRadius(outerRadius + 30)
        .startAngle((_d, i) => (i * (2 * Math.PI)) / numberOfMetrics)
        .endAngle((_d, i) => (i * (2 * Math.PI)) / numberOfMetrics)

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
                .html(`${d[0]}: ${roundToPercentage(d[1])}`)
                .style("top", `${event.pageY - 10}px`)
                .style("left", `${event.pageX + 10}px`)
        })
        .on("mouseout", function () {
            d3.select(this).attr(
                "fill",
                semiGlobalLocalSturctureColor.metricBarColor
            )
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
        .domain(d3.range(numberOfMetrics).map(String))
        .range([
            Math.PI / numberOfMetrics,
            2 * Math.PI + Math.PI / numberOfMetrics,
        ])

    const getTextAnchor = (angle: number) => {
        return "middle"
    }

    const getTransformLabel = (i: number) => {
        const angle = performanceTextScale(String(i))
        const rotate = `rotate(${(angle * 180) / Math.PI})`
        const translate = `translate(${0}, ${-outerRadius - 30})`
        return `${rotate} ${translate}`
    }

    const getTransformValue = (i: number) => {
        const angle = performanceTextScale(String(i))
        const rotate = `rotate(${(angle * 180) / Math.PI})`
        const translate = `translate(${0}, ${-outerRadius - 3})`
        return `${rotate} ${translate}`
    }

    performanceGroup
        .selectAll(".performanceLabel")
        .data((d) => Object.entries(d.localMetric))
        .join("text")
        .attr("class", "performanceLabel font-serif")
        .attr("text-anchor", (_d, i) =>
            getTextAnchor(performanceTextScale(String(i)))
        )
        .attr("fill", semiGlobalLocalSturctureColor.textColor)
        .attr("stroke", "none")
        .attr("font-size", "1.2rem")
        .attr("transform", (_d, i) => getTransformLabel(i))
        .text((d) => d[0].charAt(0).toUpperCase() + d[0].slice(1))

    performanceGroup
        .selectAll(".performanceText")
        .data((d) => Object.entries(d.localMetric))
        .join("text")
        .attr("class", "performanceText")
        .attr("text-anchor", (_d, i) =>
            getTextAnchor(performanceTextScale(String(i)))
        )
        .attr("transform", (_d, i) => getTransformValue(i))
        .attr("fill", semiGlobalLocalSturctureColor.textColor)
        .attr("stroke", "none")
        .attr("font-size", "1rem")
        .text((d) => roundToPercentage(d[1]))

    const arc = d3
        .arc()
        .innerRadius((d) => barScale(Math.min(Number(d), 0)))
        .outerRadius((d) => barScale(Math.max(Number(d), 0)))
        .startAngle((_d, i) => barIndexScale(i))
        .endAngle((_d, i) => barIndexScale(i) + barIndexScale.bandwidth())
        .padAngle(1.5 / innerRadius)
        .padRadius(innerRadius)

    node.selectAll(".bar")
        .data((d) => d.localFlatness)
        .join("path")
        .attr("class", "bar")
        .attr("data-index", (_d, i) => i + 1)
        .attr("fill", (d) =>
            d > 0 ? semiGlobalLocalSturctureColor.radioBarColor : "#cdcdcd"
        )
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
                .html(`# ${i} Hessian Eigenvalue: ${d}`)
                .style("top", `${event.pageY - 10}px`)
                .style("left", `${event.pageX + 10}px`)
        })
        .on("mouseout", function () {
            d3.select(this).attr("fill", (d) =>
                d > 0 ? semiGlobalLocalSturctureColor.radioBarColor : "#cdcdcd"
            )
            tooltip.style("visibility", "hidden")
        })

    node.selectAll(".circle")
        .data((d) => [d])
        .join("circle")
        .attr("class", "circle")
        .attr("id", (d) => `circle-${d.modelId}-${d.modeId}`)
        .attr("r", 13)
        .attr("fill", (d) => modelColorMap[d.modelId])
        .attr("stroke", "none")
        .attr("cx", 0)
        .attr("cy", 0)
        .on("mouseover", (_event, d) => {
            svgbase.selectAll(".node").style("opacity", 0.01)
            svgbase
                .select(`#nodeGroup-${d.modelId}-${d.modeId}`)
                .style("opacity", 0.8)
                .raise()
            svgbase.selectAll(".link").style("stroke-opacity", 0)
            const sourceSelector = `[id^='link-${d.modelId}-${d.modeId}-']`
            const targetSelector = `[id$='-${d.modelId}-${d.modeId}-link']`
            svgbase
                .select(`#circle-${d.modelId}-${d.modeId}`)
                .style("cursor", "pointer")
                .style("opacity", 1)
            svgbase
                .selectAll(`${sourceSelector}, ${targetSelector}`)
                .style("stroke-opacity", 1)
                .raise()
        })
        .on("mouseout", (_event, d) => {
            svgbase.selectAll(".node").style("opacity", 1)
            svgbase.selectAll(".link").style("stroke-opacity", 1)
            svgbase
                .select(`#${d.modelId}-${d.modeId}`)
                .style("cursor", "default")
        })
        .on("click", (_event, d) => {
            svgbase.selectAll(".node").style("opacity", 1)
            svgbase.selectAll(".link").style("stroke-opacity", 1)
            svgbase
                .selectAll(`[id^=outerRing-${d.modelId}]`)
                .attr("stroke-width", 1)
                .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)
            updateSelectedModelIdModeId(
                modelIdIndex,
                `${d.modelId}-${d.modeId}`
            )
        })

    const zoom = d3
        .zoom()
        .on("zoom", (event) =>
            svgbase.select(".zoom-container").attr("transform", event.transform)
        )
        .extent([
            [0, 0],
            [width, height],
        ])
        .scaleExtent([0.5, 8])

    // Center the graph visualization with a fixed value
    const centerX = width / 2
    const centerY = height / 2

    svgbase
        .call(zoom)
        .call(zoom.transform, d3.zoomIdentity.scale(1))
        .transition()
        .duration(0)
        .call(
            zoom.transform,
            d3.zoomIdentity.scale(0.52).translate(centerX, centerY - 100)
        )

    // Render modelMetaData as a compact information card
    const infoCard = svgbase
        .append("foreignObject")
        .attr("x", 10)
        .attr("y", height - 160)
        .attr("width", width - 20)
        .attr("height", 160)
        .append("xhtml:div")
        .style("background", "white")
        .style("border", "1px solid black")
        .style("padding", "5px")
        .style("font-size", "12px")
        .style("color", "black")
        .style("overflow", "hidden")
        .style("text-overflow", "ellipsis")
        .style("white-space", "pre-wrap")
        .style("word-wrap", "break-word")
        .html(
            Object.entries(modelMetaData)
                .map(([key, value]) => `<b>${key}:</b> ${value}`)
                .join("<br>")
        )

    // Add label "Global Structure" to the top left of the svg
    if (svgbase.select(".global-structure-label").empty()) {
        svgbase
            .append("text")
            .attr("class", "global-structure-label")
            .attr("x", 10)
            .attr("y", 20)
            .attr("font-size", "14px")
            .attr("font-weight", "normal")
            .attr("fill", "black")
            .text("Global Structure")
    }
}

export default function GlobalCore({
    width,
    height,
    data,
    selectedCheckPointIdList,
    updateSelectedModelIdModeId,
    modelId,
    modelIdIndex,
    modelMetaData,
}: GlobalCoreProp): React.JSX.Element {
    const svg = React.useRef<SVGSVGElement>(null)
    const wraperRef = React.useRef<HTMLDivElement>(null)

    React.useEffect(() => {
        const updateChart = () => {
            const divElement = wraperRef.current
            if (divElement) {
                const width = divElement.clientWidth
                const height = divElement.clientHeight
                d3.select(svg.current)
                    .attr("width", width)
                    .attr("height", height)
            }
        }

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
            modelIdIndex,
            modelMetaData
        )
    }, [
        data,
        width,
        height,
        selectedCheckPointIdList,
        updateSelectedModelIdModeId,
        modelId,
        modelIdIndex,
        modelMetaData,
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
