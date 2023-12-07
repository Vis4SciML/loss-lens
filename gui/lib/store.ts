import { atom } from "jotai"

import { Graph, Link, Node, SupplyPath } from "@/types/rsdn"

function getDisconnectedSubgraphs(nodes: Node[], links: Link[]) {
  const adjacencyList = new Map()
  const visited = new Set()
  const subgraphs = []

  // Step 1: Create adjacency list
  for (let node of nodes) {
    adjacencyList.set(node.id, [])
  }
  for (let link of links) {
    const source = link.source.id
    const target = link.target.id
    adjacencyList.get(source).push(target)
    adjacencyList.get(target).push(source)
  }

  // Step 2: DFS traversal helper function
  function dfs(nodeID: string, visitedNodes: Set<string>) {
    visited.add(nodeID)
    visitedNodes.add(nodeID)

    for (let neighbor of adjacencyList.get(nodeID)) {
      if (!visited.has(neighbor)) {
        dfs(neighbor, visitedNodes)
      }
    }
  }

  // Step 4: Perform DFS traversal for each unvisited node
  for (let node of nodes) {
    if (!visited.has(node.id)) {
      const visitedNodes = new Set<string>()
      dfs(node.id, visitedNodes)
      subgraphs.push(visitedNodes)
    }
  }

  return subgraphs
}

export const graphAtom = atom<Graph[] | Promise<Graph[]> | null>(null)

export const loadGraphAtom = atom(null, async (_get, set) => {
  const promise = fetchSupplyChainData().then((data) => {
    const graphs = getDisconnectedSubgraphs(data.nodes, data.links)
    return graphs.map((graph) => {
      return {
        nodes: data.nodes.filter((node: Node) => graph.has(node.id)),
        links: data.links
          .filter(
            (link: Link) =>
              graph.has(link.source.id) && graph.has(link.target.id)
          )
          .map((link: Link) => {
            const source = link.source.id
            const target = link.target.id
            return {
              id: `${source}-${target}`,
              source,
              target,
            }
          }),
      }
    })
  })
  set(graphAtom, promise)
})

export const subgraphAtom = atom<Graph | null>(null)

export const fetchSubgraphAtom = atom(
  (get) => get(subgraphAtom),
  async (get, set) => {
    const graphs = await get(graphAtom)
    const selectedGraph = get(selectedGraphAtom)
    console.log("selectedGraph in fetchSubgraphAtom")
    console.log(selectedGraph)
    if (selectedGraph !== null && graphs !== null) {
      set(subgraphAtom, graphs[selectedGraph])
    }
  }
)

export const graphKeysAtom = atom(async (get) => {
  const graphs = await get(graphAtom)
  return graphs?.map(
    (graph, index) =>
      "graph " +
      index +
      " (" +
      graph.nodes.length +
      " nodes, " +
      graph.links.length +
      " links)"
  )
})

export const showNetworkAtom = atom(false)

export const showSupplyPathAtom = atom(false)

export const selectedGraphAtom = atom<number | null>(null)

export const showSupplyChainGraphAtom = atom(false)

export const supplyPathAtom = atom<SupplyPath | null>(null)

export const fetchSupplyPathAtom = atom(
  (get) => get(supplyPathAtom),
  async (get, set) => {
    const graphs = await get(graphAtom)
    const selectedGraph = get(selectedGraphAtom)
    if (selectedGraph !== null && graphs !== null) {
      const subgraph = graphs[selectedGraph]
      const adjacencyList = new Map()

      console.log("subgraph")
      console.log(subgraph)

      subgraph?.links.forEach((link: Link) => {
        const source = link.source
        const target = link.target
        if (adjacencyList.has(source)) {
          adjacencyList.get(source).push(target)
        } else {
          adjacencyList.set(source, [target])
        }
      })

      console.log("adjacencyList")
      console.log(adjacencyList)

      let startNode = null
      let maxNeighbors = 0

      adjacencyList.forEach((neis, key) => {
        if (neis.length > maxNeighbors) {
          startNode = key
          maxNeighbors = neis.length
        }
      })

      console.log("startNode")
      console.log(startNode)

      const data = generatePathStructure(subgraph, startNode)

      set(supplyPathAtom, data)
    }
  }
)

function generatePathStructure(data: Graph, startNode: string) {
  interface TempPathStructure {
    [key: number]: Link[]
  }
  interface TempNodeStructure {
    [key: number]: {
      [key: string]: {
        [key: string]: {
          cy: number
        }
      }
    }
  }

  const pathStructure: TempPathStructure = {}
  const nodeStructure: TempNodeStructure = {}

  const edges = data.links.map((link: Link) => {
    return [link.source, link.target]
  })

  const adjacencyList = new Map()
  edges.forEach((edge) => {
    const source = edge[0]
    const target = edge[1]
    if (adjacencyList.has(source)) {
      adjacencyList.get(source).push(target)
    } else {
      adjacencyList.set(source, [target])
    }
  })

  nodeStructure[0] = {}
  nodeStructure[0][startNode] = {}
  const queue = edges
    .filter((edge) => edge[0] === startNode)
    .map((edge) => {
      return [edge, 1]
    })
  const visitedEdges = new Set()
  queue.forEach((q) => {
    const edge = q[0]
    visitedEdges.add(edge[0] + "-" + edge[1])
  })

  while (queue.length > 0) {
    const [edge, layer] = queue.shift()

    if (pathStructure[layer]) {
      pathStructure[layer].push(edge)
    } else {
      pathStructure[layer] = [edge]
    }
    if (nodeStructure[layer]) {
      if (!nodeStructure[layer][edge[1]]) {
        nodeStructure[layer][edge[1]] = {}
      }
    } else {
      nodeStructure[layer] = {}
      nodeStructure[layer][edge[1]] = {}
    }

    const neighbors = adjacencyList.get(edge[1])
    if (neighbors) {
      const neighborsEdges = neighbors.map((neighbor) => {
        return [edge[1], neighbor]
      })
      neighborsEdges
        .filter((edge) => !visitedEdges.has(edge[0] + "-" + edge[1]))
        .forEach((edge) => {
          queue.push([edge, layer + 1])
          visitedEdges.add(edge[0] + "-" + edge[1])
        })
    }
  }

  const nodes = Object.keys(nodeStructure).map((key, layerID) => {
    const subNodesPerLayer = Object.keys(nodeStructure[key])
    return subNodesPerLayer.map((node, index) => {
      nodeStructure[layerID][node]["cy"] = index * 100 + 50

      return {
        id: node,
        name: node,
        layer: layerID,
        cy: index * 100 + 50,
      }
    })
  })
  // console.group("pathStructure")
  // console.log(pathStructure)
  // console.groupEnd()
  //
  // console.group("nodeStructure")
  // console.log(nodeStructure)
  // console.groupEnd()

  const paths = Object.keys(pathStructure).map((layerID: number) => {
    const edges = pathStructure[layerID]
    return edges.map((edge) => {
      return {
        x1: (layerID - 1) * 200 + 50,
        x2: layerID * 200 + 50,
        y1: nodeStructure[layerID - 1][edge[0]].cy,
        y2: nodeStructure[layerID][edge[1]].cy,
      }
    })
  })

  return {
    pathStructure: paths,
    nodeStructure: nodes,
  }
}

async function fetchSupplyChainData() {
  // const response = await fetch("./bbn-data-converted.json")
  const response = await fetch("./copperOneAway.json")
  return await response.json()
}

export const activatedTabAtom = atom("force-layout")
export const getActivatedTabAtom = atom((get) => get(activatedTabAtom))
