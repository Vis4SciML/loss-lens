import sys
import os
import numpy as np
from tqdm import tqdm
from typing import List
import csv


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from database.db_util import *

# from update_db import *


def update_mode_losslandscape(case_id: str, model_id: str, mode_id: str, losslandscape: List[List[float]]):
    max_value = max([max(row) for row in losslandscape])
    min_value = min([min(row) for row in losslandscape])

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


def update_mode_merge_tree(case_id: str, model_id: str, mode_id: str, merge_tree: Dict[str, any]):
    if not dbExists():
        createDB()

    if not collectionExists(MERGE_TREE):
        createCollection(MERGE_TREE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}

    record = merge_tree
    addOrUpdateDocument(MERGE_TREE, query, record)


def process_loss_landscapes():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + "/temp_data/loss_landscapes_npy")

    for file_name in file_names:
        # if file_name ends with .npy, then read it
        if file_name.endswith(".npy"):
            if file_name.startswith("resnet20") and "distance_0.5" in file_name:

                file_path = current_dir + "/temp_data/loss_landscapes_npy/" + file_name
                data = np.load(file_path, allow_pickle=True)
                data = data.tolist()
                file_name_array = file_name.split("_")
                seed = file_name_array[7]

                residual = file_name_array[5]

                case_id = "resnet20"
                if residual == "True":
                    model_id = "cifar10_resnet20"
                else:
                    model_id = "cifar10_resnet20_no_skip"

                mode_id = seed

                # print(case_id, model_id, mode_id)
                update_mode_losslandscape(case_id, model_id, mode_id, data)

            elif file_name.startswith("VIT"):
                file_path = current_dir + "/temp_data/loss_landscapes_npy/" + file_name
                data = np.load(file_path, allow_pickle=True)
                data = data.tolist()
                file_name_array = file_name.split("_")
                seed = file_name_array[3]
                aug = file_name_array[5]

                case_id = "vit"
                if aug == "{00}":
                    model_id = "cifar10_vit"
                else:
                    model_id = "cifar10_augvit"

                mode_id = seed

                update_mode_losslandscape(case_id, model_id, mode_id, data)

            elif file_name.startswith("pretrained") and file_name.endswith(".npy"):
                file_path = current_dir + "/temp_data/loss_landscapes_npy/" + file_name
                data = np.load(file_path, allow_pickle=True)
                print("data shape: ", data.shape)
                print(data.shape)
                data = np.reshape(data, (40, 40))
                data = data.tolist()

                file_name_array = file_name.split("_")
                seed = file_name_array[10][4:]

                beta = file_name_array[4]

                case_id = "pinn"
                if beta == "beta1.0":
                    model_id = "pinn_convection_beta1"
                else:
                    model_id = "pinn_convection_beta50"

                mode_id = seed

                print(case_id, model_id, mode_id)
                update_mode_losslandscape(case_id, model_id, mode_id, data)
            
            else:
                print("File name not recognized")
        

def process_merge_trees_planar(input_file: str) -> dict:
    # lists used to store the data
    pointsx = []
    pointsy = []
    pointsz = []
    nodeID = []
    branchID = []
    start = []
    end = []

    # initialize some global variables
    root_x = 0

    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pointsx.append(float(row['Points:0']))
            pointsy.append(float(row['Points:1']))
            pointsz.append(float(row['Points:2']))
            nodeID.append(int(row['NodeId']))
            branchID.append(int(row['BranchNodeID']))
            # find the start point of each branch
            if int(row['BranchNodeID']) == 0:
                if int(row['NodeId']) == 0:
                    root_x = float(row['Points:0'])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row['Points:0']) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
                end[i] = 1
                end[j] = 1

    # verify that the start and end points are correct
    temp_structure = []
    for i in range(len(start)):
        t = {'start': start[i], 'end': end[i], 'x': pointsx[i], 'y': pointsy[i], 'z': pointsz[i], 'nodeID': nodeID[i], 'branchID': branchID[i]}
        temp_structure.append(t)

    nodes = []
    for item in temp_structure:
        nodes.append({
            "id": item["nodeID"],
            "x": item["x"],
            "y": item["y"],
        })

    edges = []
    branch = {}

    for i in range(len(temp_structure)):
        item_id = temp_structure[i]["branchID"]
        if item_id not in branch:
            branch[item_id] = []
        branch[item_id].append(temp_structure[i])

    for key in branch:
        nodes = branch[key]
        for i in range(len(nodes) - 1):
            for j in range(i + 1, len(nodes) - 1):
                if i != j and (nodes[i]['x'] == nodes[j]['x'] or nodes[i]['y'] == nodes[j]['y']):
                    edges.append({
                        "sourceX": nodes[i]['x'],
                        "sourceY": nodes[i]['y'],
                        "targetX": nodes[j]['x'],
                        "targetY": nodes[j]['y'],
                    })

    res =  {
        "nodes": nodes,
        "edges": edges,
    }

    # print(res)
    return res



def process_merge_tree(input_file: str) -> dict:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

       "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Points:0","Points:1","Points:2"
        0,4.6e+02,33,0,3,2,33,0,0
        1,1.5,99,0,5,4,19,2,0
        2,0.95,378,0,4,3,18,9,0

    """
    # process merge_tree object here
    pointsx = []
    pointsy = []
    pointsz = []
    nodeID = []
    branchID = []
    start = []
    end = []

    # initialize some global variables
    root_x = 0

    # open the csv file and load the data into the lists
    with open(input_file, newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row['Points:0']))
            pointsy.append(float(row['Points:1']))
            pointsz.append(float(row['Points:2']))
            
            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row['NodeId']))
            
            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row['CriticalType'])) 
            
            # find the start point of each branch
            if int(row['CriticalType']) == 0:
                if int(row['NodeId']) == 0:
                    root_x = float(row['Points:0'])
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(1)
            else:
                if float(row['Points:0']) == root_x:
                    start.append(1)
                    end.append(0)
                else:
                    start.append(0)
                    end.append(0)

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
                end[i] = 1
                end[j] = 1


    # convert to representation for the database
    nodes = [
        {"x": pointsx[i], "y": pointsy[i], "id": nodeID[i]} 
        for i in range(len(start))
    ]

    # TODO: how are edges defined? we need to find source, target then get X,Y of each
    edges = [
        {
            "sourceX": pointsx[i],
            "sourceY": pointsy[i],
            "targetX": 0, 
            "targetY": 0,
        }
        for i in range(len(start))
    ]
    merge_tree = {"nodes": nodes, "edges": edges}

   
    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')
    
    return merge_tree

def process_merge_trees():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + "/temp_data/loss_landscapes_MT_PD")

    for file_name in file_names:
        if file_name.endswith(".csv"):
            if file_name.startswith("VIT") and file_name.endswith("MergeTreePlanar.csv"):
                file_path = current_dir + "/temp_data/loss_landscapes_MT_PD/" + file_name
                merge_tree = process_merge_trees_planar(file_path)

                file_name_array = file_name.split("_")
                seed = file_name_array[3]
                aug = file_name_array[5]

                case_id = "vit"
                if aug == "{00}":
                    model_id = "cifar10_vit"
                else:
                    model_id = "cifar10_augvit"

                mode_id = seed

                # print(case_id, model_id, mode_id)
                # print(merge_tree)
                update_mode_merge_tree(case_id, model_id, mode_id, merge_tree)

            elif file_name.startswith("resnet20") and file_name.endswith("MergeTreePlanar.csv") and "distance_0.5" in file_name:
                file_path = current_dir + "/temp_data/loss_landscapes_MT_PD/" + file_name
                merge_tree = process_merge_trees_planar(file_path)
                # print(merge_tree)

                file_name_array = file_name.split("_")
                seed = file_name_array[7]

                residual = file_name_array[5]

                case_id = "resnet20"
                if residual == "True":
                    model_id = "cifar10_resnet20"
                else:
                    model_id = "cifar10_resnet20_no_skip"

                mode_id = seed

                #print(case_id, model_id, mode_id)
                # print(merge_tree)

                update_mode_merge_tree(case_id, model_id, mode_id, merge_tree)
            elif file_name.startswith("pretrained") and file_name.endswith("MergeTreePlanar.csv"):
                file_path = current_dir + "/temp_data/loss_landscapes_MT_PD/" + file_name
                merge_tree = process_merge_trees_planar(file_path)
                # print(merge_tree)

                file_name_array = file_name.split("_")
                seed = file_name_array[10][4:]

                beta = file_name_array[4]

                case_id = "pinn"
                if beta == "beta1.0":
                    model_id = "pinn_convection_beta1"
                else:
                    model_id = "pinn_convection_beta50"

                mode_id = seed

                print(case_id, model_id, mode_id)
                # print(merge_tree)

                update_mode_merge_tree(case_id, model_id, mode_id, merge_tree)



def process_persistence_barcode(input_file: str) -> list:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

        "ttkVertexScalarField","CriticalType","Coordinates:0","Coordinates:1","Coordinates:2","Points:0","Points:1","Points:2"
        896,0,16,22,0,0.017,0.017,0
        1599,3,39,39,0,0.017,1.6e+03,0
        1056,0,16,26,0,0.028,0.028,0

    """

    # process persistence_barcode object here
    points_0 = []
    points_1 = []
    points_2 = []
    nodeID = []
    
    # load coordinates from csv file
    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            points_0.append(float(row["Points:0"]))
            points_1.append(float(row["Points:1"]))
            points_2.append(float(row["Points:2"]))
            
            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row["Point ID"]))
            nodeID.append(int(row["ttkVertexScalarField"]))


    # convert to representation for the database
    # TODO: not sure what x, y0, y1 correspond to
    persistence_barcode = [
        {"y0": points_0[i], "y1": points_1[i], "x": points_2[i]}
        for i in range(len(nodeID))
    ]

    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')

    return persistence_barcode


def update_mode_persistence_barcode(case_id: str, model_id: str, mode_id: str, persistence_barcode: dict):
    if not dbExists():
        createDB()

    if not collectionExists(PERSISTENCE_BARCODE):
        createCollection(PERSISTENCE_BARCODE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}
    record = {"edges": persistence_barcode}
    addOrUpdateDocument(PERSISTENCE_BARCODE, query, record)

def process_persistence_diagrams():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_names = os.listdir(current_dir + "/temp_data/loss_landscapes_MT_PD")

    for file_name in file_names:
        if file_name.endswith(".csv"):
            if file_name.startswith("VIT") and file_name.endswith("PersistenceDiagram.csv"):
                file_path = current_dir + "/temp_data/loss_landscapes_MT_PD/" + file_name
                pd = process_persistence_barcode(file_path)

                file_name_array = file_name.split("_")
                seed = file_name_array[3]
                aug = file_name_array[5]

                case_id = "vit"
                if aug == "{00}":
                    model_id = "cifar10_vit"
                else:
                    model_id = "cifar10_augvit"

                mode_id = seed

                update_mode_persistence_barcode(case_id, model_id, mode_id, pd)

            elif file_name.startswith("resnet20") and file_name.endswith("PersistenceDiagram.csv") and "distance_0.5" in file_name:
                file_path = current_dir + "/temp_data/loss_landscapes_MT_PD/" + file_name
                pd = process_persistence_barcode(file_path)
                # print(merge_tree)

                file_name_array = file_name.split("_")
                seed = file_name_array[7]

                residual = file_name_array[5]

                case_id = "resnet20"
                if residual == "True":
                    model_id = "cifar10_resnet20"
                else:
                    model_id = "cifar10_resnet20_no_skip"

                mode_id = seed

                update_mode_persistence_barcode(case_id, model_id, mode_id, pd)

            elif file_name.startswith("pretrained") and file_name.endswith("PersistenceDiagram.csv"):
                file_path = current_dir + "/temp_data/loss_landscapes_MT_PD/" + file_name
                pd = process_persistence_barcode(file_path)

                file_name_array = file_name.split("_")
                seed = file_name_array[10][4:]

                beta = file_name_array[4]

                case_id = "pinn"
                if beta == "beta1.0":
                    model_id = "pinn_convection_beta1"
                else:
                    model_id = "pinn_convection_beta50"

                mode_id = seed

                print(case_id, model_id, mode_id)
                update_mode_persistence_barcode(case_id, model_id, mode_id, pd)



def update_single_landscape():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "resnet20_batch_norm_True_residual_False_seed_2023_net_hessian_False_batch_size_512_distance_0.5_steps_40_norm_layer_random_normal.npy"
    file_path = current_dir + "/temp_data/loss_landscapes_npy/" + file_name
    data = np.load(file_path, allow_pickle=True)
    data = data.tolist()
    file_name_array = file_name.split("_")
    seed = file_name_array[7]

    residual = file_name_array[5]

    case_id = "resnet20"
    if residual == "True":
        model_id = "cifar10_resnet20"
    else:
        model_id = "cifar10_resnet20_no_skip"

    mode_id = seed

    # print(case_id, model_id, mode_id)
    update_mode_losslandscape(case_id, model_id, mode_id, data)



if __name__ == "__main__":
    update_single_landscape()
    # process_loss_landscapes()
    # process_merge_trees()
    # process_persistence_diagrams()




