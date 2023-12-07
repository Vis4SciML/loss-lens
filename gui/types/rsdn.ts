export interface Node {
  id: string
  type: string
  properties: {
    city: string
    country: string
    county: string
    duns: number
    facility_name: string
    federal_facility: string
    latitude: number
    longitude: number
    naics: string
    region: number
    state: string
    street: string
    streat_number: string
    zip_code: number
  }
  x?: number
  y?: number
}

interface Contract {
  id: string
  properties: {
    award_date: number
    contract_number: string
    derive_method: string
    good: {
      federal_supply_chain_classification: number
      id: string
      item_name: string
      naics: string
      nsn: string
      security_classification: string
    }
    good_id: string
    id: string
    net_price: number
    order_quantity: number
    part_number: string
    purchase_order_number: number
    purchase_order_item_number: number
    source_file: string
    standard_unit_price: number
  }
}

export interface Link {
  id: string
  properties: {
    weight: number
    contracts: Contract[]
  }
  source: {
    id: string
    x?: number
    y?: number
  }
  target: {
    id: string
    x?: number
    y?: number
  }
}

export interface Graph {
  nodes: Node[]
  links: Link[]
}

export interface NodeStructure {
  id: string
  name: string
  layer: number
  cy: number
}

export interface PathStructure {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface SupplyPath {
  nodeStructure: NodeStructure[][]
  pathStructure: PathStructure[][]
}
