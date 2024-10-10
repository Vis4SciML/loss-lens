import sys
import os
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from database.db_util import *


def update_mode_losslandscape(
    case_id: str, model_id: str, mode_id: str, losslandscape: List[List[float]]
):
    max_value = max(max(row) for row in losslandscape)
    min_value = min(min(row) for row in losslandscape)

    if not dbExists():
        createDB()

    if not collectionExists(LOSS_LANDSCAPE):
        createCollection(LOSS_LANDSCAPE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}
    record = {"grid": losslandscape}
    addOrUpdateDocument(LOSS_LANDSCAPE, query, record)

    boundary_query = {"caseId": case_id}
    boundary_record = getDocument(MODEL_META_DATA, boundary_query) or {
        "caseId": case_id
    }

    if "lossBounds" not in boundary_record:
        boundary_record["lossBounds"] = {
            "upperBound": max_value,
            "lowerBound": min_value,
        }
    else:
        boundary_record["lossBounds"]["upperBound"] = max(
            max_value, boundary_record["lossBounds"]["upperBound"]
        )
        boundary_record["lossBounds"]["lowerBound"] = min(
            min_value, boundary_record["lossBounds"]["lowerBound"]
        )

    addOrUpdateDocument(MODEL_META_DATA, boundary_query, boundary_record)


def update_mode_merge_tree(
    case_id: str, model_id: str, mode_id: str, merge_tree: Dict[str, any]
):
    if not dbExists():
        createDB()

    if not collectionExists(MERGE_TREE):
        createCollection(MERGE_TREE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}
    addOrUpdateDocument(MERGE_TREE, query, merge_tree)


def process_loss_landscapes():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(os.path.join(current_dir, "../data/loss_landscape_files"))

    for file_name in file_names:
        if file_name.endswith(".npy"):
            file_path = os.path.join(
                current_dir, "../data/loss_landscape_files", file_name
            )
            data = np.load(file_path, allow_pickle=True).tolist()
            file_name_array = file_name.split("_")

            if file_name.startswith("resnet20") and "distance_0.5" in file_name:
                seed = file_name_array[7]
                residual = file_name_array[5]
                case_id = "resnet20"
                model_id = (
                    "cifar10_resnet20"
                    if residual == "True"
                    else "cifar10_resnet20_no_skip"
                )
                mode_id = seed
                update_mode_losslandscape(case_id, model_id, mode_id, data)

            elif file_name.startswith("VIT"):
                seed = file_name_array[3]
                aug = file_name_array[5]
                case_id = "vit"
                model_id = "cifar10_vit" if aug == "{00}" else "cifar10_augvit"
                mode_id = seed
                update_mode_losslandscape(case_id, model_id, mode_id, data)

            elif file_name.startswith("pretrained"):
                data = np.reshape(data, (40, 40)).tolist()
                seed = file_name_array[10][4:]
                beta = file_name_array[4]
                case_id = "pinn"
                model_id = (
                    "pinn_convection_beta1"
                    if beta == "beta1.0"
                    else "pinn_convection_beta50"
                )
                mode_id = seed
                update_mode_losslandscape(case_id, model_id, mode_id, data)
            elif "PINN" in file_name:
                data = np.reshape(data, (21, 21)).tolist()
                seed = file_name_array[4]
                beta = file_name_array[3]
                case_id = "pinn"
                model_id = (
                    "PINN_convection_beta_1.0"
                    if beta == "1.0"
                    else "PINN_convection_beta_50.0"
                )
                mode_id = seed
                update_mode_losslandscape(case_id, model_id, mode_id, data)
            else:
                print("File name not recognized")


def process_merge_trees_planar(input_file: str) -> dict:
    pointsx, pointsy, pointsz, nodeID, branchID, start, end = [], [], [], [], [], [], []
    root_x = 0

    with open(input_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pointsx.append(float(row["Points:0"]))
            pointsy.append(float(row["Points:1"]))
            pointsz.append(float(row["Points:2"]))
            nodeID.append(int(row["NodeId"]))
            branchID.append(int(row["BranchNodeID"]))
            if int(row["BranchNodeID"]) == 0:
                if int(row["NodeId"]) == 0:
                    root_x = float(row["Points:0"])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row["Points:0"]) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)

    for i in range(len(start)):
        this_x, this_y, this_z = pointsx[i], pointsy[i], pointsz[i]
        for j in range(len(start)):
            if (
                this_x == pointsx[j]
                and this_y == pointsy[j]
                and this_z == pointsz[j]
                and i != j
            ):
                end[i] = 1
                end[j] = 1

    temp_structure = [
        {
            "start": start[i],
            "end": end[i],
            "x": pointsx[i],
            "y": pointsy[i],
            "z": pointsz[i],
            "nodeID": nodeID[i],
            "branchID": branchID[i],
        }
        for i in range(len(start))
    ]

    nodes = [
        {"id": item["nodeID"], "x": item["x"], "y": item["y"]}
        for item in temp_structure
    ]

    edges = []
    branch = {}
    for item in temp_structure:
        item_id = item["branchID"]
        if item_id not in branch:
            branch[item_id] = []
        branch[item_id].append(item)

    for key in branch:
        nodes = branch[key]
        for i in range(len(nodes) - 1):
            for j in range(i + 1, len(nodes) - 1):
                if i != j and (
                    nodes[i]["x"] == nodes[j]["x"] or nodes[i]["y"] == nodes[j]["y"]
                ):
                    edges.append(
                        {
                            "sourceX": nodes[i]["x"],
                            "sourceY": nodes[i]["y"],
                            "targetX": nodes[j]["x"],
                            "targetY": nodes[j]["y"],
                        }
                    )

    return {"nodes": nodes, "edges": edges}


def process_merge_tree(input_file: str) -> dict:
    pointsx, pointsy, pointsz, nodeID, branchID, start, end = [], [], [], [], [], [], []
    root_x = 0

    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row["Points:0"]))
            pointsy.append(float(row["Points:1"]))
            pointsz.append(float(row["Points:2"]))
            nodeID.append(int(row["NodeId"]))
            branchID.append(int(row["CriticalType"]))
            if int(row["CriticalType"]) == 0:
                if int(row["NodeId"]) == 0:
                    root_x = float(row["Points:0"])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row["Points:0"]) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)

    for i in range(len(start)):
        this_x, this_y, this_z = pointsx[i], pointsy[i], pointsz[i]
        for j in range(len(start)):
            if (
                this_x == pointsx[j]
                and this_y == pointsy[j]
                and this_z == pointsz[j]
                and i != j
            ):
                end[i] = 1
                end[j] = 1

    nodes = [
        {"x": pointsx[i], "y": pointsy[i], "id": nodeID[i]} for i in range(len(start))
    ]
    edges = [
        {"sourceX": pointsx[i], "sourceY": pointsy[i], "targetX": 0, "targetY": 0}
        for i in range(len(start))
    ]
    return {"nodes": nodes, "edges": edges}


def process_merge_trees():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(os.path.join(current_dir, "../data/paraview_files"))

    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_path = os.path.join(current_dir, "../data/paraview_files", file_name)
            if file_name.startswith("VIT") and file_name.endswith(
                "MergeTreePlanar.csv"
            ):
                merge_tree = process_merge_trees_planar(file_path)
                file_name_array = file_name.split("_")
                seed = file_name_array[3]
                aug = file_name_array[5]
                case_id = "vit"
                model_id = "cifar10_vit" if aug == "{00}" else "cifar10_augvit"
                mode_id = seed
                update_mode_merge_tree(case_id, model_id, mode_id, merge_tree)

            elif (
                file_name.startswith("resnet20")
                and file_name.endswith("MergeTreePlanar.csv")
                and "distance_0.5" in file_name
            ):
                merge_tree = process_merge_trees_planar(file_path)
                file_name_array = file_name.split("_")
                seed = file_name_array[7]
                residual = file_name_array[5]
                case_id = "resnet20"
                model_id = (
                    "cifar10_resnet20"
                    if residual == "True"
                    else "cifar10_resnet20_no_skip"
                )
                mode_id = seed
                update_mode_merge_tree(case_id, model_id, mode_id, merge_tree)

            elif file_name.startswith("pretrained") and file_name.endswith(
                "MergeTreePlanar.csv"
            ):
                merge_tree = process_merge_trees_planar(file_path)
                file_name_array = file_name.split("_")
                seed = file_name_array[10][4:]
                beta = file_name_array[4]
                case_id = "pinn"
                model_id = (
                    "pinn_convection_beta1"
                    if beta == "beta1.0"
                    else "pinn_convection_beta50"
                )
                mode_id = seed
                update_mode_merge_tree(case_id, model_id, mode_id, merge_tree)
            elif file_name.startswith("PINN") and file_name.endswith(
                "MergeTreePlanar.csv"
            ):
                merge_tree = process_merge_trees_planar(file_path)
                file_name_array = file_name.split("_")
                seed = file_name_array[4]
                beta = file_name_array[3]
                case_id = "pinn"
                model_id = (
                    "PINN_convection_beta_1.0"
                    if beta == "1.0"
                    else "PINN_convection_beta_50.0"
                )
                mode_id = seed
                update_mode_merge_tree(case_id, model_id, mode_id, merge_tree)


def process_persistence_barcode(input_file: str) -> list:
    points_0, points_1, points_2, nodeID = [], [], [], []

    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            points_0.append(float(row["Points:0"]))
            points_1.append(float(row["Points:1"]))
            points_2.append(float(row["Points:2"]))
            nodeID.append(int(row["ttkVertexScalarField"]))

    return [
        {"y0": points_0[i], "y1": points_1[i], "x": points_2[i]}
        for i in range(len(nodeID))
    ]


def update_mode_persistence_barcode(
    case_id: str, model_id: str, mode_id: str, persistence_barcode: list
):
    if not dbExists():
        createDB()

    if not collectionExists(PERSISTENCE_BARCODE):
        createCollection(PERSISTENCE_BARCODE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}
    record = {"edges": persistence_barcode}
    addOrUpdateDocument(PERSISTENCE_BARCODE, query, record)


def process_persistence_diagrams():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(os.path.join(current_dir, "../data/paraview_files"))

    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_path = os.path.join(current_dir, "../data/paraview_files", file_name)
            if file_name.startswith("VIT") and file_name.endswith(
                "PersistenceDiagram.csv"
            ):
                pd = process_persistence_barcode(file_path)
                file_name_array = file_name.split("_")
                seed = file_name_array[3]
                aug = file_name_array[5]
                case_id = "vit"
                model_id = "cifar10_vit" if aug == "{00}" else "cifar10_augvit"
                mode_id = seed
                update_mode_persistence_barcode(case_id, model_id, mode_id, pd)

            elif (
                file_name.startswith("resnet20")
                and file_name.endswith("PersistenceDiagram.csv")
                and "distance_0.5" in file_name
            ):
                pd = process_persistence_barcode(file_path)
                file_name_array = file_name.split("_")
                seed = file_name_array[7]
                residual = file_name_array[5]
                case_id = "resnet20"
                model_id = (
                    "cifar10_resnet20"
                    if residual == "True"
                    else "cifar10_resnet20_no_skip"
                )
                mode_id = seed
                update_mode_persistence_barcode(case_id, model_id, mode_id, pd)

            elif file_name.startswith("pretrained") and file_name.endswith(
                "PersistenceDiagram.csv"
            ):
                pd = process_persistence_barcode(file_path)
                file_name_array = file_name.split("_")
                seed = file_name_array[10][4:]
                beta = file_name_array[4]
                case_id = "pinn"
                model_id = (
                    "pinn_convection_beta1"
                    if beta == "beta1.0"
                    else "pinn_convection_beta50"
                )
                mode_id = seed
                update_mode_persistence_barcode(case_id, model_id, mode_id, pd)
            elif file_name.startswith("PINN") and file_name.endswith(
                "PersistenceDiagram.csv"
            ):
                pd = process_persistence_barcode(file_path)
                file_name_array = file_name.split("_")
                seed = file_name_array[4]
                beta = file_name_array[3]
                case_id = "pinn"
                model_id = (
                    "PINN_convection_beta_1.0"
                    if beta == "1.0"
                    else "PINN_convection_beta_50.0"
                )
                mode_id = seed
                update_mode_persistence_barcode(case_id, model_id, mode_id, pd)


def update_single_landscape():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "resnet20_batch_norm_True_residual_False_seed_2023_net_hessian_False_batch_size_512_distance_0.5_steps_40_norm_layer_random_normal.npy"
    file_path = os.path.join(current_dir, "temp_data/loss_landscapes_npy", file_name)
    data = np.load(file_path, allow_pickle=True).tolist()
    file_name_array = file_name.split("_")
    seed = file_name_array[7]
    residual = file_name_array[5]
    case_id = "resnet20"
    model_id = "cifar10_resnet20" if residual == "True" else "cifar10_resnet20_no_skip"
    mode_id = seed
    update_mode_losslandscape(case_id, model_id, mode_id, data)


if __name__ == "__main__":
    # update_single_landscape()
    # process_loss_landscapes()
    process_merge_trees()
    process_persistence_diagrams()
