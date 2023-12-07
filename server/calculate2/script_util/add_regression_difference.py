'''
This script is used to load pinn model and save the regression difference 
to the database


'''


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

def load_pinn_mode(mode_name: str, model_list: list):
    if mode_name.startswith(model_list[0]):
        mode_path = (
            parent_dir
            + "/trained_models/"
            + model_list[0]
            + "/"
            + mode_name
        )
    elif mode_name.startswith(model_list[1]):
        mode_path = (
            parent_dir
            + "/trained_models/"
            + model_list[1]
            + "/"
            + mode_name
        )
    
    model = torch.load(mode_path)
    return model





def add_pinn_regression_difference():
    # get all records from layer-similarity
    if not dbExists():
        print("No database found")
        return

    if not collectionExists(REGRESSION_DIFFERENCE):
        createCollection(REGRESSION_DIFFERENCE)
    
    # training data?
    xgrid = 64    
    x = np.linspace(0, 2*np.pi, xgrid, endpoint=False).reshape(-1, 1) # not inclusive
    t = np.linspace(0, 1, 100).reshape(-1, 1)
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None].astype(np.float), T.flatten()[:, None].astype(np.float))) # all the x,t "test" data

    # remove initial and boundaty data from X_star
    t_noinitial = t[1:]
    # remove boundary at x=0
    x_noboundary = x[1:]
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)
    X_f_train = sample_random(X_star_noinitial_noboundary, 100)

    beta = 1
    # training labels?
    u_vals = convection_diffusion("sin(x)", 1.0, beta, 0, xgrid, 100)
    G = np.full(X_f_train.shape[0], float(0))

    u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)

    
    model_list = ["pinn_convection_beta1", "pinn_convection_beta50"]

    file_names_1 = os.listdir(parent_dir + "/trained_models/" + model_list[0])
    file_names_2 = os.listdir(parent_dir + "/trained_models/" + model_list[1])

    file_names = file_names_1 + file_names_2
    print(file_names)

    for i in range(len(file_names)):
        file_name_1 = file_names[i]
        if not file_name_1.startswith(model_list[0]) and not file_name_1.startswith(model_list[1]):
            continue
        for j in range(len(file_names)):
            file_name_2 = file_names[j]
            if not file_name_2.startswith(model_list[1]) and not file_name_2.startswith(model_list[0]):
                continue
            if file_name_1 == file_name_2:
                continue
            print("file_name_1: ", file_name_1)
            print("file_name_2: ", file_name_2)

            mode1 = load_pinn_mode(file_name_1, model_list)
            mode2 = load_pinn_mode(file_name_2, model_list)
            u_pred1 = mode1.predict(X_star)
            u_pred2 = mode2.predict(X_star)
            # print(u_pred1.shape)
            # print(u_pred2.shape)
            # print(u_star.shape)
            grid1 = np.hstack((u_star, u_pred1))
            grid2 = np.hstack((u_star, u_pred2))

            data = {
                "caseId": "pinn",
                "modePairId": file_name_1.split(".")[0] + "_" + file_name_2.split(".")[0],
                "modesId": [file_name_1.split(".")[0], file_name_2.split(".")[0]],
                "grid": {
                    "a": grid1.tolist(),
                    "b": grid2.tolist()
                }
            }

            query = {"caseId": "pinn", "modePairId": data["modePairId"]}

            addOrUpdateDocument(REGRESSION_DIFFERENCE, query, data)





            




    





if __name__ == "__main__":
    add_pinn_regression_difference()
