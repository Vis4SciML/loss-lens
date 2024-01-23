import random
import time
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.types import Device
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import numpy as np



def compute_mode_performance(model_id: str, mode_id: str) -> Dict[str, float]:
    performance = {}

    performance["accuracy"] = random.random()
    performance["precision"] = random.random()
    performance["recall"] = random.random()
    performance["f1"] = random.random()

    return performance


def compute_mode_hessian(model_id: str, mode_id: str) -> List[List[float]]:
    hessian = [random.random() for i in range(10)]

    return hessian


def compute_mode_losslandscape(model_id: str, mode_id: str) -> List[List[float]]:
    losslandscape = [[random.random() for i in range(10)] for j in range(10)]
    max_value = max([max(row) for row in losslandscape])
    min_value = min([min(row) for row in losslandscape])

    return losslandscape, max_value, min_value


def compute_mode_merge_tree(model_id: str, mode_id: str) -> List[Dict[str, float]]:
    nodes = [{"x": random.random(), "y": random.random(), "id": i} for i in range(100)]
    edges = [
        {
            "sourceX": random.random(),
            "sourceY": random.random(),
            "targetX": random.random(),
            "targetY": random.random(),
        }
        for i in range(100)
    ]
    merge_tree = {"nodes": nodes, "edges": edges}

    return merge_tree


def compute_mode_persistence_barcode(
    model_id: str, mode_id: str
) -> List[Dict[str, float]]:
    persistence_barcode = [
        {"x": random.random(), "y0": random.random(), "y1": random.random()}
        for i in range(100)
    ]

    return persistence_barcode


def compute_cka_similarity(
    model0_id: str, model1_id: str, mode0_id: str, mode1_id: str
) -> float:
    cka_similarity = random.random()

    return cka_similarity


def compute_layer_similarity(
    model0_id: str, model1_id: str, mode0_id: str, mode1_id: str
) -> List[List[float]]:
    layer_similarity = [[random.random() for i in range(100)] for j in range(100)]

    return layer_similarity


def compute_mode_connectivity(model_id: str, mode0_id: str, mode1_id: str) -> float:
    connectivity = random.random()

    return connectivity


def compute_confusion_matrix(
    model0_id: str, model1_id: str, mode0_id: str, mode1_id: str
) -> List[List[float]]:
    confusion_matrix = [
        {
            "tp": [random.random(), random.random()],
            "fp": [random.random(), random.random()],
            "tn": [random.random(), random.random()],
            "fn": [random.random(), random.random()],
        }
        for j in range(10)
    ]

    class_names = ["class" + str(i) for i in range(10)]
    return confusion_matrix, class_names


def compute_position(distances) -> List[float]:
    res = {}
    for distance in distances:
        id = distance["model0Id"] + "-" + distance["mode0Id"]
        res[id] = {
            "x": random.random(),
            "y": random.random(),
        }
        id = distance["model1Id"] + "-" + distance["mode1Id"]
        res[id] = {
            "x": random.random(),
            "y": random.random(),
        }

    return res
