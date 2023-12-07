# get all records from layer-similarity, and for each record, get the labels from the corresponding model

import sys
import os
import numpy as np
from tqdm import tqdm
from typing import List
from vit_pytorch import ViT
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_dir + "/training_scripts/")
sys.path.append(parent_dir + "/training_scripts/pinn/pbc_examples/")

from database.db_util import *

from pinn.pbc_examples.choose_optimizer import *
from training_scripts.RESNET20 import resnet
from pinn.pbc_examples.net_pbc import * 
from pinn.pbc_examples.utils import *
from pinn.pbc_examples.systems_pbc import *
from pinn.pyhessian import hessian_pinn

def add_labels_to_layer_similarity(case_id: str):
    # get all records from layer-similarity

    if not dbExists():
        print("No database found")
        return

    if not collectionExists(LAYER_SIMILARITY):
        print("No layer-similarity collection found")
        return
    
    query = { "caseId": case_id }

    records = getDocuments(LAYER_SIMILARITY, query)
    records = list(records)

    # get model
    if case_id == "resnet20":
        model = resnet(num_classes=10,
               depth=20,
               residual_not=True,
               batch_norm_not=True)
        model_no_skip = resnet(num_classes=10, 
                depth=20,
                residual_not=False,
                batch_norm_not=True)

        layer_names_params = [name for name, _ in model.named_modules()]
        layer_names_no_skip_params = [name for name, _ in model_no_skip.named_modules()]
        for record in records:
            modePairId = record["modePairId"]
            model_index = modePairId.find("resnet20_no_skip")
            
            print("modePairId: ", modePairId)
            if model_index == -1:
                # model is resnet20
                print("two resnet20")
                record["xLabels"] = layer_names_params
                record["yLabels"] = layer_names_params

            elif model_index == 8:
                # do not know if it is 2 noskip or 1 noskip 
                print("do not know if it is 2 noskip or 1 noskip")
                modePairId = modePairId[9:]
                print("modePairId: ", modePairId)
                model_index = modePairId.find("resnet20_no_skip")
                if model_index == -1:
                    print("one noskip and one skip")
                    record["xLabels"] = layer_names_no_skip_params
                    record["yLabels"] = layer_names_params
                else: 
                    print("two noskip")
                    record["xLabels"] = layer_names_no_skip_params
                    record["yLabels"] = layer_names_no_skip_params
            else:
                print("one skip and one noskip")
                record["xLabels"] = layer_names_params
                record["yLabels"] = layer_names_no_skip_params

            
            addOrUpdateDocument(LAYER_SIMILARITY, {"caseId": case_id, "modePairId": record["modePairId"]}, record)


    elif case_id == "vit":
        model = ViT(image_size = 32, patch_size = 4, num_classes = 10, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)
        layer_names_params = [name for name, _ in model.named_modules()]
        for record in records:
            record["xLabels"] = layer_names_params
            record["yLabels"] = layer_names_params
            addOrUpdateDocument(LAYER_SIMILARITY, {"caseId": case_id, "modePairId": record["modePairId"]}, record)
    
    elif case_id == "pinn":
        mode_path = (
            parent_dir
            + "/trained_models/"
            + "pinn_convection_beta1"
            + "/"
            + "pinn_convection_beta1"
            + "_123"
            + ".pt"
        )
        model = torch.load(mode_path)
        model = model.dnn
        layer_names_params = [name for name, _ in model.named_modules()]
        print(layer_names_params)
        for record in records:
            record["xLabels"] = layer_names_params
            record["yLabels"] = layer_names_params
            addOrUpdateDocument(LAYER_SIMILARITY, {"caseId": case_id, "modePairId": record["modePairId"]}, record)

    else:

        print("Invalid case id")
        return



    




if __name__ == "__main__":
    case_id = "vit"
    # case_id = "pinn"
    add_labels_to_layer_similarity(case_id)
