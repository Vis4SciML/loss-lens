"use client"

import * as React from "react"
import * as d3 from "d3"

export interface SNode {
  id: string
  name: string
}

export interface SLink {
  source: string
  target: string
}

export interface SGraph {
  nodes: SNode[]
  links: SLink[]
}

export interface TangledTreeLayerItem {
  id: string
  parents?: string[]
  children?: string[]
  level?: number
  links?: string[]
}

export interface TangledTreeDataType {
  data: TangledTreeLayerItem[][] | null
}

function constructTangleLayout(levels: TangledTreeLayerItem[][], options = {}) {
  if (!levels || levels.length == 0) {
    return
  }
  // precompute level depth
  levels.forEach((l, i) => l.forEach((n) => (n.level = i)))

  console.log("levels")
  console.log(levels)
  var nodes = levels.reduce((a, x) => a.concat(x), [])
  var nodes_index = {}
  nodes.forEach((d) => (nodes_index[d.id] = d))

  console.log("nodes")
  console.log(nodes)
  // objectification
  nodes.forEach((d) => {
    d.parents = (d.parents === undefined ? [] : d.parents).map(
      (p) => nodes_index[p]
    )
  })

  // precompute bundles
  levels.forEach((l, i) => {
    var index = {}
    l.forEach((n) => {
      if (n.parents.length == 0) {
        return
      }

      console.log("n.parents")
      console.log(n)
      console.log(n.parents)
      var id = n.parents
        .map((d) => d.id)
        .sort()
        .join("-X-")
      if (id in index) {
        index[id].parents = index[id].parents.concat(n.parents)
      } else {
        index[id] = {
          id: id,
          parents: n.parents.slice(),
          level: i,
          span: i - d3.min(n.parents, (p) => p.level),
        }
      }
      n.bundle = index[id]
    })
    l.bundles = Object.keys(index).map((k) => index[k])
    l.bundles.forEach((b, i) => (b.i = i))
  })

  var links = []
  nodes.forEach((d) => {
    d.parents.forEach((p) =>
      links.push({ source: d, bundle: d.bundle, target: p })
    )
  })

  var bundles = levels.reduce((a, x) => a.concat(x.bundles), [])

  // reverse pointer from parent to bundles
  bundles.forEach((b) =>
    b.parents.forEach((p) => {
      if (p.bundles_index === undefined) {
        p.bundles_index = {}
      }
      if (!(b.id in p.bundles_index)) {
        p.bundles_index[b.id] = []
      }
      p.bundles_index[b.id].push(b)
    })
  )

  nodes.forEach((n) => {
    if (n.bundles_index !== undefined) {
      n.bundles = Object.keys(n.bundles_index).map((k) => n.bundles_index[k])
    } else {
      n.bundles_index = {}
      n.bundles = []
    }
    n.bundles.sort((a, b) =>
      d3.descending(
        d3.max(a, (d) => d.span),
        d3.max(b, (d) => d.span)
      )
    )
    n.bundles.forEach((b, i) => (b.i = i))
  })

  links.forEach((l) => {
    if (l.bundle.links === undefined) {
      l.bundle.links = []
    }
    l.bundle.links.push(l)
  })

  // layout
  const padding = 8
  const node_height = 22
  const node_width = 70
  const bundle_width = 14
  const level_y_padding = 16
  const metro_d = 4
  const min_family_height = 22

  options.c ||= 16
  const c = options.c
  options.bigc ||= node_width + c

  nodes.forEach(
    (n) => (n.height = (Math.max(1, n.bundles.length) - 1) * metro_d)
  )

  var x_offset = padding
  var y_offset = padding
  levels.forEach((l) => {
    x_offset += l.bundles.length * bundle_width
    y_offset += level_y_padding
    l.forEach((n, i) => {
      n.x = n.level * node_width + x_offset
      n.y = node_height + y_offset + n.height / 2

      y_offset += node_height + n.height
    })
  })

  var i = 0
  levels.forEach((l) => {
    l.bundles.forEach((b) => {
      b.x =
        d3.max(b.parents, (d) => d.x) +
        node_width +
        (l.bundles.length - 1 - b.i) * bundle_width
      b.y = i * node_height
    })
    i += l.length
  })

  links.forEach((l) => {
    l.xt = l.target.x
    l.yt =
      l.target.y +
      l.target.bundles_index[l.bundle.id].i * metro_d -
      (l.target.bundles.length * metro_d) / 2 +
      metro_d / 2
    l.xb = l.bundle.x
    l.yb = l.bundle.y
    l.xs = l.source.x
    l.ys = l.source.y
  })

  // compress vertical space
  var y_negative_offset = 0
  levels.forEach((l) => {
    y_negative_offset +=
      -min_family_height +
        d3.min(l.bundles, (b) =>
          d3.min(b.links, (link) => link.ys - 2 * c - (link.yt + c))
        ) || 0
    l.forEach((n) => (n.y -= y_negative_offset))
  })

  // very ugly, I know
  links.forEach((l) => {
    l.yt =
      l.target.y +
      l.target.bundles_index[l.bundle.id].i * metro_d -
      (l.target.bundles.length * metro_d) / 2 +
      metro_d / 2
    l.ys = l.source.y
    l.c1 =
      l.source.level - l.target.level > 1
        ? Math.min(options.bigc, l.xb - l.xt, l.yb - l.yt) - c
        : c
    l.c2 = c
  })

  var layout = {
    width: d3.max(nodes, (n) => n.x) + node_width + 2 * padding,
    height: d3.max(nodes, (n) => n.y) + node_height / 2 + 2 * padding,
    node_height,
    node_width,
    bundle_width,
    level_y_padding,
    metro_d,
  }

  return { levels, nodes, nodes_index, links, bundles, layout }
}

function render(
  svgRef: React.RefObject<SVGSVGElement>,
  data: TangledTreeLayerItem[][]
) {
  const svgbase = d3.select(svgRef.current)
  console.log("data in render: ")
  console.log(data)

  const tangleLayout = constructTangleLayout(data)
  svgbase.attr("width", tangleLayout.layout.width)
  svgbase.attr("height", tangleLayout.layout.height)

  const paths = svgbase
    .selectAll(".link")
    .data(tangleLayout.bundles)
    .join("path")
    .attr("class", "link")
    .attr("d", (b, i) => {
      let d = b.links
        .map(
          (l) => `
      M${l.xt} ${l.yt}
      L${l.xb - l.c1} ${l.yt}
      A${l.c1} ${l.c1} 90 0 1 ${l.xb} ${l.yt + l.c1}
      L${l.xb} ${l.ys - l.c2}
      A${l.c2} ${l.c2} 90 0 0 ${l.xb + l.c2} ${l.ys}
      L${l.xs} ${l.ys}`
        )
        .join("")
      return d
    })
    .attr("stroke", "gray")
    .attr("stroke-width", "2")
    .attr("fill", "none")

  const circles = svgbase
    .selectAll(".circle")
    .data(tangleLayout.nodes)
    .join("path")
    .attr("class", "circle")
    .attr("d", (n) => {
      // console.log(n)
      return `M${n.x} ${n.y - n.height / 2} L${n.x} ${n.y + n.height / 2}`
    })
    .attr("stroke", "gray")
    .attr("stroke-width", "8")
    .attr("stroke-linecap", "round")
    .attr("fill", "white")

  const circleInners = svgbase
    .selectAll(".circleInner")
    .data(tangleLayout.nodes)
    .join("path")
    .attr("class", "circleInner")
    .attr("d", (n) => {
      // console.log(n)
      return `M${n.x} ${n.y - n.height / 2} L${n.x} ${n.y + n.height / 2}`
    })
    .attr("stroke", "white")
    .attr("stroke-width", "4")
    .attr("stroke-linecap", "round")
    .attr("fill", "none")

  const texts = svgbase
    .selectAll(".text")
    .data(tangleLayout.nodes)
    .join("text")
    .attr("class", "text")
    .attr("x", (n) => n.x + 4)
    .attr("y", (n) => n.y - n.height / 2 - 4)
    .attr("font-size", "10px")
    .text((n) => n.id)
}

export default function TangledTreeCore({
  data,
}: TangledTreeDataType): React.JSX.Element {
  const svg = React.useRef<SVGSVGElement>(null)

  React.useEffect(() => {
    if (!data) return
    console.log("rendering")
    console.log(data)
    render(svg, data)
  }, [data])

  return (
    <svg ref={svg}>
      <g className="circle-group" />
    </svg>
  )
}
