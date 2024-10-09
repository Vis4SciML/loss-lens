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
    showPerformance: boolean
    showHessian: boolean
    showPerformanceLabels: boolean
    showModelInfo: boolean
    mcFilterRange: [number, number]
}

function idWrapper(id: string): string {
    return id.replace(/\./g, "")
}

function render(
    svgRef: React.RefObject<SVGSVGElement>,
    wraperRef: React.RefObject<HTMLDivElement>,
    data: SemiGlobalLocalStructure,
    selectedCheckPointIdList: string[],
    updateSelectedModelIdModeId: (index: number, id: string) => void,
    modelId: string,
    modelIdIndex: number,
    modelMetaData: any,
    showPerformance: boolean,
    showHessian: boolean,
    showPerformanceLabels: boolean,
    showModelInfo: boolean,
    width: number,
    height: number,
    mcFilterRange: [number, number]
) {
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
        .range([1, 30])

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
            link.source.modelId === modelId &&
            link.target.modelId === modelId &&
            link.weight >= mcFilterRange[0] &&
            link.weight <= mcFilterRange[1] // Filter based on mcFilterRange
    )

    const nodes = wholeNodes.filter((node) => node.modelId === modelId)

    const positionNode = (d: ModeNode, newScale = 1) =>
        `translate(${xScale(d.x) * newScale},${yScale(d.y) * newScale})`

    const generateCurvePath = (edge, newScale = 1) => {
        const [scaledSourceX, scaledSourceY] = [
            xScale(edge.source.x) * newScale,
            yScale(edge.source.y) * newScale,
        ]
        const [scaledTargetX, scaledTargetY] = [
            xScale(edge.target.x) * newScale,
            yScale(edge.target.y) * newScale,
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
                `link-${idWrapper(d.source.modelId)}-${
                    d.source.modeId
                }-${idWrapper(d.target.modelId)}-${d.target.modeId}-link`
        )
        .attr("stroke-width", (d) => linkThickness(d.weight))
        .attr("stroke-opacity", 0.2)
        .attr("fill", "none")
        .attr("d", (d) => generateCurvePath(d))
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

    const innerRadius = Math.min(w, h) * 0.025
    const outerRadius = Math.min(w, h) * 0.045

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

    const performanceBarScale = d3
        .scaleLinear()
        .domain([0, 1])
        .range([0, Math.PI / 2])

    const metricArc = d3
        .arc()
        .innerRadius(outerRadius * 1.002) // outerRadius + 3 proportional
        .outerRadius(outerRadius * 1.5) // outerRadius + 20 proportional
        .startAngle((_d, i) => (i * (2 * Math.PI)) / numberOfMetrics)
        .endAngle(
            (d, i) =>
                (i * (2 * Math.PI)) / numberOfMetrics +
                performanceBarScale(d[1])
        )

    const metricArcLine = d3
        .arc()
        .innerRadius(outerRadius * 1.002) // outerRadius + 3 proportional
        .outerRadius(outerRadius * 1.5) // outerRadius + 20 proportional
        .startAngle((_d, i) => (i * (2 * Math.PI)) / numberOfMetrics)
        .endAngle((_d, i) => (i * (2 * Math.PI)) / numberOfMetrics)

    const arc = d3
        .arc()
        .innerRadius((d) => barScale(Math.min(Number(d), 0)))
        .outerRadius((d) => barScale(Math.max(Number(d), 0)))
        .startAngle((_d, i) => barIndexScale(String(i)) ?? 0)
        .endAngle(
            (_d, i) => (barIndexScale(i) ?? 0) + barIndexScale.bandwidth()
        )
        .padAngle(1.5 / innerRadius)
        .padRadius(innerRadius)

    const node = svgbase
        .selectAll(".nodes")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1)
        .selectAll(".node")
        .data(nodes)
        .join("g")
        .attr("class", "node")
        .attr("id", (d) => `nodeGroup-${idWrapper(d.modelId)}-${d.modeId}`)
        .attr("transform", (d) => positionNode(d))

    const localPropertyGroup = node
        .selectAll(".localPropertyGroup")
        .data((d) => [d])
        .join("g")
        .attr("class", "localPropertyGroup")

    const performanceGroup = localPropertyGroup
        .selectAll(".performanceGroup")
        .data((d) => [d])
        .join("g")
        .attr("class", "performanceGroup")
        .style("opacity", showPerformance ? 1 : 0)

    const hessianGroup = localPropertyGroup
        .selectAll(".hessianGroup")
        .data((d) => [d])
        .join("g")
        .attr("class", "hessianGroup")
        .style("opacity", showHessian ? 1 : 0)

    const outerRing = performanceGroup
        .selectAll(".outerRing")
        .data((d) => [d])
        .join("circle")
        .attr("class", "outerRing")
        .attr("id", (d) => `outerRing-${idWrapper(d.modelId)}-${d.modeId}`)
        .attr("r", outerRadius * 1.5)
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

    const getTransformLabel = (i: number, newScale = 1) => {
        const angle = performanceTextScale(String(i))
        const rotate = `rotate(${(angle * 180) / Math.PI})`
        const translate = `translate(${0}, ${-outerRadius * 1.6 * newScale})`
        return `${rotate} ${translate}`
    }

    const getTransformValue = (i: number, newScale = 1) => {
        const angle = performanceTextScale(String(i))
        const rotate = `rotate(${(angle * 180) / Math.PI})`
        const translate = `translate(${0}, ${-outerRadius * 1.1 * newScale})`
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
        .attr("font-size", `${innerRadius * 0.5}px`)
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
        .attr("font-size", `${innerRadius * 0.48}px`)
        .text((d) => roundToPercentage(d[1]))

    hessianGroup
        .selectAll(".middleRing")
        .data((d) => [d])
        .join("circle")
        .attr("class", "middleRing")
        .attr("id", (d) => `middleRing-${idWrapper(d.modelId)}-${d.modeId}`)
        .attr("r", outerRadius * 1.002)
        .attr("fill", semiGlobalLocalSturctureColor.itemInnerBackgroundColor)
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)

    hessianGroup
        .selectAll(".hessianBar")
        .data((d) => {
            const hessian = d.localFlatness
            return d.localFlatness
        })
        .join("path")
        .attr("class", "hessianBar")
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

    const modelGroup = localPropertyGroup
        .selectAll(".modelGroup")
        .data((d) => [d])
        .join("g")
        .attr("class", "modelGroup")

    modelGroup
        .selectAll(".innerRing")
        .data((d) => [d])
        .join("circle")
        .attr("class", "innerRing")
        .attr("id", (d) => `innerRing-${idWrapper(d.modelId)}-${d.modeId}`)
        .attr("r", barScale(0))
        .attr("fill", semiGlobalLocalSturctureColor.itemInnerBackgroundColor)
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)
        .attr("stroke-width", 0.5)

    modelGroup
        .selectAll(".circle")
        .data((d) => [d])
        .join("circle")
        .attr("class", "circle")
        .attr("id", (d) => `circle-${idWrapper(d.modelId)}-${d.modeId}`)
        .attr("r", innerRadius * 0.52)
        .attr("fill", (d) => modelColorMap[d.modelId])
        .attr("stroke", "none")
        .attr("cx", 0)
        .attr("cy", 0)
        .on("mouseover", (_event, d) => {
            svgbase.selectAll(".node").style("opacity", 0)
            svgbase
                .select(`#nodeGroup-${idWrapper(d.modelId)}-${d.modeId}`)
                .style("opacity", 1)
                .raise()
            svgbase.selectAll(".link").style("stroke-opacity", 0)
            const sourceSelector = `[id^='link-${idWrapper(d.modelId)}-${
                d.modeId
            }-']`
            const targetSelector = `[id$='-${idWrapper(d.modelId)}-${
                d.modeId
            }-link']`
            svgbase
                .select(`#circle-${idWrapper(d.modelId)}-${d.modeId}`)
                .style("cursor", "pointer")
                .style("opacity", 1)
            svgbase
                .selectAll(`${sourceSelector}, ${targetSelector}`)
                .style("stroke-opacity", 1)
                .raise()
        })
        .on("mouseout", (_event, d) => {
            svgbase.selectAll(".node").style("opacity", 1)
            svgbase.selectAll(".link").style("stroke-opacity", 0.2)
            svgbase
                .select(`#circle-${idWrapper(d.modelId)}-${d.modeId}`)
                .style("cursor", "default")
        })
        .on("click", (_event, d) => {
            svgbase
                .selectAll(".outerRing")
                .attr("stroke-width", 1)
                .attr("stroke", semiGlobalLocalSturctureColor.strokeColor)
            svgbase.selectAll(".node").style("opacity", 1)
            svgbase.selectAll(".link").style("stroke-opacity", 1)
            svgbase
                .selectAll(`#outerRing-${idWrapper(d.modelId)}-${d.modeId}`)
                .attr("stroke-width", 4)
                .attr("stroke", modelColorMap[d.modelId])
            updateSelectedModelIdModeId(
                modelIdIndex,
                `${d.modelId}-${d.modeId}`
            )
        })

    const zoom = d3
        .zoom()
        .on("zoom", (event) => {
            svgbase.select(".zoom-container").attr("transform", event.transform)
            const newScale = event.transform.k * 4 // Increase the magnitude of the zoom scale
            svgbase.selectAll(".node").attr("transform", (d) => {
                return positionNode(d, newScale)
            })
            svgbase
                .selectAll(".link")
                .attr("d", (d) => generateCurvePath(d, newScale))
                .attr("stroke-width", (d) => linkThickness(d.weight) / newScale) // Scale the thickness of the edges
            // svgbase
            //     .selectAll(".circle")
            //     .attr("r", (innerRadius * 0.52) / newScale) // Scale the size of the nodes
            // svgbase.selectAll(".innerRing").attr("r", barScale(0) / newScale) // Scale the size of the inner ring
            // svgbase
            //     .selectAll(".outerRing")
            //     .attr("r", (outerRadius * 1.5) / newScale) // Scale the size of the outer ring
            // svgbase
            //     .selectAll(".middleRing")
            //     .attr("r", (outerRadius * 1.002) / newScale) // Scale the size of the middle ring
            // svgbase
            //     .selectAll(".performanceLabel")
            //     .attr("font-size", `${(innerRadius * 0.8) / newScale}px`) // Scale the size of the performance labels
            // svgbase
            //     .selectAll(".performanceText")
            //     .attr("font-size", `${(innerRadius * 0.8) / newScale}px`) // Scale the size of the performance text
        })
        .extent([
            [0, 0],
            [width, height],
        ])
        .scaleExtent([0.1, 8])

    svgbase.call(zoom)

    // Center the graph visualization with a fixed value
    const centerX = width / 2
    const centerY = height / 2

    svgbase
        .transition()
        .duration(0)
        .call(
            zoom.transform,
            d3.zoomIdentity.scale(0.5).translate(centerX, centerY)
        )

    // Render modelMetaData as a compact information card
    const infoCard = svgbase
        .append("foreignObject")
        .attr("x", 10)
        .attr("y", height - 160)
        .attr("width", width - 20)
        .attr("height", 160)
        .style("display", showModelInfo ? "block" : "none") // Toggle visibility
        .append("xhtml:div")
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

    const performanceLabels = performanceGroup
        .selectAll(".performanceLabel")
        .style("opacity", showPerformanceLabels ? 1 : 0)
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
    showPerformance,
    showHessian,
    showPerformanceLabels,
    showModelInfo,
    mcFilterRange,
}: GlobalCoreProp): React.JSX.Element {
    const svg = React.useRef<SVGSVGElement>(null)
    const wraperRef = React.useRef<HTMLDivElement>(null)
    const [isInitialized, setIsInitialized] = React.useState(false)

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
        if (!data || isInitialized) return
        const clonedData = JSON.parse(JSON.stringify(data))
        render(
            svg,
            wraperRef,
            clonedData,
            selectedCheckPointIdList,
            updateSelectedModelIdModeId,
            modelId,
            modelIdIndex,
            modelMetaData,
            showPerformance,
            showHessian,
            showPerformanceLabels,
            showModelInfo,
            width,
            height,
            mcFilterRange
        )
        setIsInitialized(true)
    }, [data, isInitialized])

    React.useEffect(() => {
        if (!isInitialized) return
        const svgElement = d3.select(svg.current)
        svgElement
            .selectAll(".performanceGroup")
            .style("opacity", showPerformance ? 1 : 0)
        svgElement
            .selectAll(".hessianGroup")
            .style("opacity", showHessian ? 1 : 0)
        svgElement
            .selectAll(".performanceLabel")
            .style("opacity", showPerformanceLabels ? 1 : 0)
        svgElement
            .selectAll("foreignObject")
            .style("display", showModelInfo ? "block" : "none")
        svgElement
            .selectAll(".link")
            .style("display", (d) =>
                d.weight >= mcFilterRange[0] && d.weight <= mcFilterRange[1]
                    ? "block"
                    : "none"
            )
    }, [
        showPerformance,
        showHessian,
        showPerformanceLabels,
        showModelInfo,
        isInitialized,
        mcFilterRange,
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
