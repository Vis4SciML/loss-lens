import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from script_util.update_db import *

case_id = "vit"

meta_datas = [
    {
        "modelId": "vit",
        "modelName": "vit",
        "modelDescription": "2-layer MLP, 512 hidden units, batch size 128, 10 epochs",
        "modelDataset": "MNIST",
        "datasetId": "mnist",
        "modelDatasetDescription": "Original MNIST dataset, 60k training images, 10k test images",
    },
    {
        "modelId": "augvit",
        "modelName": "augvit",
        "modelDescription": "2-layer MLP, 512 hidden units, batch size 128, 10 epochs",
        "modelDataset": "MNIST + Corrupted MNIST images",
        "datasetId": "augmnist",
        "modelDatasetDescription": "Original MNIST dataset, 60k training images, 10k test images, Corrupted ",
    },
]


sample_modes = [
    {
        "caseId": "vit",
        "modelId": "vit",
        "modeId": "123",
    },
    {
        "caseId": "vit",
        "modelId": "vit",
        "modeId": "456",
    },
    {
        "caseId": "vit",
        "modelId": "vit",
        "modeId": "789",
    },
    {
        "caseId": "vit",
        "modelId": "vit",
        "modeId": "111",
    },
    {
        "caseId": "vit",
        "modelId": "augvit",
        "modeId": "321",
    },
    {
        "caseId": "vit",
        "modelId": "augvit",
        "modeId": "654",
    },
    {
        "caseId": "vit",
        "modelId": "augvit",
        "modeId": "987",
    },
]

for sample_mode in sample_modes:
    case_id = sample_mode["caseId"]
    model_id = sample_mode["modelId"]
    mode_id = sample_mode["modeId"]
    update_mode_performance(case_id, model_id, mode_id)
    update_mode_hessian(case_id, model_id, mode_id)
    update_mode_losslandscape(case_id, model_id, mode_id)
    update_mode_merge_tree(case_id, model_id, mode_id)
    update_mode_persistence_barcode(case_id, model_id, mode_id)
    update_mode_layer_similarity(case_id, model_id, mode_id)
    update_mode_connectivity(case_id, model_id, mode_id)
    update_mode_confusion_matrix(case_id, model_id, mode_id)
    update_mode_cka_similarity(case_id, model_id, mode_id)


for meta_data in meta_datas:
    model_id = meta_data["modelId"]
    update_model_meta_data(case_id, model_id, meta_data)
