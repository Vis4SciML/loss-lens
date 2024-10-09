import sys
import os
import csv
from typing import Dict

from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from database.db_util import *
from script_util.update_db import update_mode_connectivity


def process_mc_scores(file_path: str, case_id: str, model_id: str):
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        with tqdm(
            total=sum(1 for _ in reader),
            desc="Processing MC Scores",
            unit="row",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total:0,} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)
            for row in reader:
                mode_id1 = row["source"]
                mode_id2 = row["target"]
                connectivity_score = float(row["mc"])
                # Assuming the CSV has columns 'mode_id1', 'mode_id2' and 'connectivity_score'
                mode_id1 = mode_id1.zfill(3)
                mode_id2 = mode_id2.zfill(3)
                update_mode_connectivity(
                    case_id, model_id, mode_id1, mode_id2, connectivity_score
                )
                pbar.update(1)


def update_mode_connectivity(
    case_id: str, model_id: str, mode_id1: str, mode_id2: str, connectivity_score: float
):

    if not dbExists():
        createDB()

    if not collectionExists(SEMI_GLOBAL_LOCAL_STRUCTURE):
        createCollection(SEMI_GLOBAL_LOCAL_STRUCTURE)

    node_query = {"caseId": case_id}
    node_record = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query)
    if node_record is None:
        return

    nodes = node_record["nodes"]

    # turn nodes into a dictionary
    nodes_dict = {node["modelId"] + "_" + node["modeId"]: node for node in nodes}

    if "links" not in node_record:
        edges = []
    else:
        edges = node_record["links"]

    edges_dict = {edge["modePairId"]: edge for edge in edges}

    mode_pair_id = model_id + "_" + mode_id1 + "_" + mode_id2
    mode_pair_id_alt = model_id + "_" + mode_id2 + "_" + mode_id1
    if mode_pair_id in edges_dict:
        source_node_id = model_id + "_" + mode_id1
        target_node_id = model_id + "_" + mode_id2
        edges_dict[mode_pair_id]["type"] = "well" if connectivity_score > 0 else "poor"
        edges_dict[mode_pair_id]["weight"] = connectivity_score
        edges_dict[mode_pair_id]["source"]["x"] = nodes_dict[source_node_id]["x"]
        edges_dict[mode_pair_id]["source"]["y"] = nodes_dict[source_node_id]["y"]
        edges_dict[mode_pair_id]["target"]["x"] = nodes_dict[target_node_id]["x"]
        edges_dict[mode_pair_id]["target"]["y"] = nodes_dict[target_node_id]["y"]
    elif mode_pair_id_alt in edges_dict:
        edges_dict[mode_pair_id_alt]["type"] = (
            "well" if connectivity_score > 0 else "poor"
        )
        edges_dict[mode_pair_id_alt]["weight"] = connectivity_score
        edges_dict[mode_pair_id_alt]["source"]["x"] = nodes_dict[source_node_id]["x"]
        edges_dict[mode_pair_id_alt]["source"]["y"] = nodes_dict[source_node_id]["y"]
        edges_dict[mode_pair_id_alt]["target"]["x"] = nodes_dict[target_node_id]["x"]
        edges_dict[mode_pair_id_alt]["target"]["y"] = nodes_dict[target_node_id]["y"]
    else:
        edge = {
            "modePairId": mode_pair_id,
            "source": {
                "modeId": mode_id1,
                "modelId": model_id,
                "x": nodes_dict[source_node_id]["x"],
                "y": nodes_dict[source_node_id]["y"],
            },
            "target": {
                "modeId": mode_id2,
                "modelId": model_id,
                "x": nodes_dict[target_node_id]["x"],
                "y": nodes_dict[target_node_id]["y"],
            },
            "type": "well" if connectivity_score > 0 else "poor",
            "weight": connectivity_score,
        }
        edges_dict[mode_pair_id] = edge

    edges = list(edges_dict.values())
    record = {"links": edges}
    addOrUpdateDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, node_query, record)


def main():
    # Define file paths
    file_paths = [
        current_dir
        + "/../data/PINN_convection_beta_1.0_lr_1.0_n_seeds_100_pairs_MC_scores.csv",
        current_dir
        + "/../data/PINN_convection_beta_50.0_lr_1.0_n_seeds_100_pairs_MC_scores.csv",
    ]

    # Process each file
    for file_path in file_paths:
        if "beta_1.0" in file_path:
            case_id = "pinn"
            model_id = "PINN_convection_beta_1.0"
        elif "beta_50.0" in file_path:
            case_id = "pinn"
            model_id = "PINN_convection_beta_50.0"

        process_mc_scores(file_path, case_id, model_id)


if __name__ == "__main__":
    main()
