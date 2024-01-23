import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from script_util.core_functions_placeholder import *
from database.db_util import *


def update_mode_performance(case_id: str, model_id: str, mode_id: str):
    performance = compute_mode_performance(model_id, mode_id)

    if not dbExists():
        createDB()

    if not collectionExists(SEMI_GLOBAL_LOCAL_STRUCTURE):
        createCollection(SEMI_GLOBAL_LOCAL_STRUCTURE)

    query = {
        "caseId": case_id,
    }

    record = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, query)
    if record is None:
        record = {
            "caseId": case_id,
        }
    if "nodes" not in record.keys():
        record["nodes"] = []

    nodes = record["nodes"]
    search_index = -1
    for node in nodes:
        if node["modelId"] == model_id and node["modeId"] == mode_id:
            search_index = nodes.index(node)
            break

    if search_index == -1:
        node = {"modelId": model_id, "modeId": mode_id, "localMetric": performance}
        nodes.append(node)
    else:
        nodes[search_index]["localMetric"] = performance

    record["nodes"] = nodes

    addOrUpdateDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, query, record)


def update_mode_hessian(case_id: str, model_id: str, mode_id: str):
    hessian = compute_mode_hessian(model_id, mode_id)
    print("hessian: ", hessian)

    if not dbExists():
        createDB()

    if not collectionExists(SEMI_GLOBAL_LOCAL_STRUCTURE):
        createCollection(SEMI_GLOBAL_LOCAL_STRUCTURE)

    query = {
        "caseId": case_id,
    }

    record = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, query)
    if record is None:
        record = {
            "caseId": case_id,
        }
    if "nodes" not in record.keys():
        record["nodes"] = []

    nodes = record["nodes"]
    search_index = -1
    for node in nodes:
        if node["modelId"] == model_id and node["modeId"] == mode_id:
            search_index = nodes.index(node)
            break

    if search_index == -1:
        node = {"modelId": model_id, "modeId": mode_id, "localMetric": hessian}
        nodes.append(node)
    else:
        nodes[search_index]["localFlatness"] = hessian

    record["nodes"] = nodes

    addOrUpdateDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, query, record)


def update_mode_losslandscape(case_id: str, model_id: str, mode_id: str):
    losslandscape, max_value, min_value = compute_mode_losslandscape(model_id, mode_id)

    if not dbExists():
        createDB()

    if not collectionExists(LOSS_LANDSCAPE):
        createCollection(LOSS_LANDSCAPE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}

    record = {"grid": losslandscape}
    addOrUpdateDocument(LOSS_LANDSCAPE, query, record)

    boundary_query = {"caseId": case_id}
    boundary_record = getDocument(MODEL_META_DATA, boundary_query)
    if boundary_record is None:
        boundary_record = {
            "caseId": case_id,
        }
    if "lossBounds" not in boundary_record:
        boundary_record["lossBounds"] = {
            "upperBound": max_value,
            "lowerBound": min_value,
        }
    else:
        boundary_record["lossBounds"]["upperBound"] = (
            max_value
            if max_value > boundary_record["lossBounds"]["upperBound"]
            else boundary_record["lossBounds"]["upperBound"]
        )
        boundary_record["lossBounds"]["lowerBound"] = (
            min_value
            if min_value < boundary_record["lossBounds"]["lowerBound"]
            else boundary_record["lossBounds"]["lowerBound"]
        )

    addOrUpdateDocument(MODEL_META_DATA, boundary_query, boundary_record)


def update_mode_merge_tree(case_id: str, model_id: str, mode_id: str):
    merge_tree = compute_mode_merge_tree(model_id, mode_id)

    if not dbExists():
        createDB()

    if not collectionExists(MERGE_TREE):
        createCollection(MERGE_TREE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}

    record = merge_tree
    addOrUpdateDocument(MERGE_TREE, query, record)


def update_mode_persistence_barcode(case_id: str, model_id: str, mode_id: str):
    persistence_barcode = compute_mode_persistence_barcode(model_id, mode_id)

    if not dbExists():
        createDB()

    if not collectionExists(PERSISTENCE_BARCODE):
        createCollection(PERSISTENCE_BARCODE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}
    record = {"edges": persistence_barcode}
    addOrUpdateDocument(PERSISTENCE_BARCODE, query, record)


def update_mode_layer_similarity(case_id: str, model_id: str, mode_id: str):
    """
    Here, we need to get all modes and computer layer similarity for each mode.
    The computed layer similarity is stored in a document, but all needs to be
    pulled out and recompute the x and y for nodes and edges.
    """
    if not dbExists():
        createDB()

    if not collectionExists(SEMI_GLOBAL_LOCAL_STRUCTURE):
        createCollection(SEMI_GLOBAL_LOCAL_STRUCTURE)

    node_query = {"caseId": case_id}
    node_record = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query)
    if node_record is None:
        return

    nodes = node_record["nodes"]

    for node in nodes:
        similarity = compute_layer_similarity(
            model0_id=model_id,
            model1_id=node["modelId"],
            mode0_id=mode_id,
            mode1_id=node["modeId"],
        )
        mode_pair_id = (
            model_id + "_" + mode_id + "_" + node["modelId"] + "_" + node["modeId"]
        )
        layer_similarity_query = {"caseId": case_id, "modePairId": mode_pair_id}
        record = {
            "caseId": case_id,
            "modePairId": mode_pair_id,
            "grid": similarity,
            "upperBound": max(max(row) for row in similarity),
            "lowerBound": min(min(row) for row in similarity),
        }

        addOrUpdateDocument(LAYER_SIMILARITY, layer_similarity_query, record)


def update_mode_connectivity(case_id: str, model_id: str, mode_id: str):
    """
    Here, we need to get all modes within the same model and compute mode connectivity
    for each mode. After that, store it in the semi-global-local-structure collection.

    """
    if not dbExists():
        createDB()

    if not collectionExists(SEMI_GLOBAL_LOCAL_STRUCTURE):
        createCollection(SEMI_GLOBAL_LOCAL_STRUCTURE)

    node_query = {"caseId": case_id}
    node_record = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query)
    if node_record is None:
        return

    nodes = node_record["nodes"]
    if "links" not in node_record:
        edges = []
    else:
        edges = node_record["links"]

    edges_dict = {edge["modePairId"]: edge for edge in edges}

    for node in nodes:
        if node["modelId"] != model_id or (
            node["modelId"] == model_id and node["modeId"] == mode_id
        ):
            continue
        connectivity = compute_mode_connectivity(
            model_id=model_id, mode0_id=mode_id, mode1_id=node["modeId"]
        )
        mode_pair_id = model_id + "_" + mode_id + "_" + node["modeId"]
        mode_pair_id_alt = model_id + "_" + node["modeId"] + "_" + mode_id
        if mode_pair_id in edges_dict:
            edges_dict[mode_pair_id]["type"] = "well" if connectivity > 0 else "poor"
            edges_dict[mode_pair_id]["weight"] = connectivity
        elif mode_pair_id_alt in edges_dict:
            edges_dict[mode_pair_id_alt]["type"] = (
                "well" if connectivity > 0 else "poor"
            )
            edges_dict[mode_pair_id_alt]["weight"] = connectivity
        else:
            edge = {
                "modePairId": mode_pair_id,
                "source": {"modeId": mode_id, "modelId": model_id},
                "target": {"modeId": node["modeId"], "modelId": node["modelId"]},
                "type": "well" if connectivity > 0 else "poor",
                "weight": connectivity,
            }
            edges_dict[mode_pair_id] = edge

    edges = list(edges_dict.values())
    record = {"links": edges}
    addOrUpdateDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query, record)


def update_mode_confusion_matrix(case_id: str, model_id: str, mode_id: str):
    """
    Here, we need to get all modes and compute mode confusion matrix
    for each mode. After that, store it in the confusion-matrix collection.

    """
    if not dbExists():
        createDB()

    if not collectionExists(SEMI_GLOBAL_LOCAL_STRUCTURE):
        createCollection(SEMI_GLOBAL_LOCAL_STRUCTURE)

    node_query = {"caseId": case_id}
    node_record = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query)
    if node_record is None:
        return

    nodes = node_record["nodes"]

    for node in nodes:
        confusion_matrix, class_names = compute_confusion_matrix(
            model0_id=model_id,
            model1_id=node["modelId"],
            mode0_id=mode_id,
            mode1_id=node["modeId"],
        )
        mode_pair_id = (
            model_id + "_" + mode_id + "_" + node["modelId"] + "_" + node["modeId"]
        )
        confusion_matrix_query = {"caseId": case_id, "modePairId": mode_pair_id}
        record = {
            "caseId": case_id,
            "modePairId": mode_pair_id,
            "grid": confusion_matrix,
            "classesName": class_names,
        }

        addOrUpdateDocument(CONFUSION_MATRIX, confusion_matrix_query, record)


def update_mode_cka_similarity(case_id: str, model_id: str, mode_id: str):
    """
    1. Get all modes
    2. Compute cka similarity for each mode
    3. Store it in the cka-similarity collection
    4. Get all cka-similarity from cka-similarity collection
    5. Recompute the position of each mode
    6. Store it in the semi-global-local-structure collection

    """
    if not dbExists():
        createDB()

    if not collectionExists(SEMI_GLOBAL_LOCAL_STRUCTURE):
        createCollection(SEMI_GLOBAL_LOCAL_STRUCTURE)

    node_query = {"caseId": case_id}
    node_record = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query)
    if node_record is None:
        return

    nodes = node_record["nodes"]

    for node in nodes:
        if model_id == node["modelId"] and mode_id == node["modeId"]:
            continue
        cka_similarity = compute_cka_similarity(
            model0_id=model_id,
            model1_id=node["modelId"],
            mode0_id=mode_id,
            mode1_id=node["modeId"],
        )
        mode_pair_id = (
            model_id + "_" + mode_id + "_" + node["modelId"] + "_" + node["modeId"]
        )
        cka_similarity_query = {
            "caseId": case_id,
            "model0Id": model_id,
            "model1Id": node["modelId"],
            "mode0Id": mode_id,
            "mode1Id": node["modeId"],
        }
        record = {
            "ckaSimilarity": cka_similarity,
        }

        addOrUpdateDocument(CKA_SIMILARITY, cka_similarity_query, record)

    distance_query = {"caseId": case_id}

    distances = getDocuments(CKA_SIMILARITY, distance_query)

    distances = list(distances)

    if len(distances) == 0:
        return

    node_positions = compute_position(distances)
    updated_nodes = []
    for node in nodes:
        new_position = node_positions[node["modelId"] + "-" + node["modeId"]]
        node["x"] = new_position["x"]
        node["y"] = new_position["y"]
        updated_nodes.append(node)

    if "links" not in node_record:
        edges = []
    edges = node_record["links"]
    updated_edges = []
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        source["x"] = node_positions[source["modelId"] + "-" + source["modeId"]]["x"]
        source["y"] = node_positions[source["modelId"] + "-" + source["modeId"]]["y"]
        target["x"] = node_positions[target["modelId"] + "-" + target["modeId"]]["x"]
        target["y"] = node_positions[target["modelId"] + "-" + target["modeId"]]["y"]
        new_edge = edge
        new_edge["source"] = source
        new_edge["target"] = target
        updated_edges.append(new_edge)

    record = {"nodes": updated_nodes, "links": updated_edges}
    addOrUpdateDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query, record)


def update_model_meta_data(case_id: str, model_id: str, meta_data: Dict[str, str]):
    if not dbExists():
        createDB()

    if not collectionExists(MODEL_META_DATA):
        createCollection(MODEL_META_DATA)

    # check how many modes are there for this model
    node_query = {"caseId": case_id}
    node_record = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query)
    if node_record is None:
        return

    nodes = node_record["nodes"]

    number_of_modes = 0
    for node in nodes:
        if node["modelId"] == model_id:
            number_of_modes += 1

    meta_data["numberOfModes"] = number_of_modes

    model_meta_data_query = {"caseId": case_id}
    model_meta_data_record = getDocument(MODEL_META_DATA, model_meta_data_query)
    if model_meta_data_record is None or "data" not in model_meta_data_record:
        model_meta_data = []
    else:
        model_meta_data = model_meta_data_record["data"]
    search_index = -1
    for i, model in enumerate(model_meta_data):
        if model["modelId"] == model_id:
            search_index = i
            break

    if search_index != -1:
        model_meta_data[search_index] = meta_data
    else:
        model_meta_data.append(meta_data)

    record = {"data": model_meta_data}
    addOrUpdateDocument(MODEL_META_DATA, model_meta_data_query, record)

    if "modelList" not in node_record:
        model_list = []
    else:
        model_list = node_record["modelList"]

    search_index = -1
    for i, model in enumerate(model_list):
        if model == model_id:
            search_index = i
            break

    if search_index != -1:
        model_list[search_index] = meta_data["modelId"]
    else:
        model_list.append(meta_data["modelId"])

    record = {"modelList": model_list}
    addOrUpdateDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query, record)
