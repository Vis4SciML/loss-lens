
###############################################################################
# imports
###############################################################################

import os
import sys
from typing import Dict, List, Optional, Union
import subprocess
import csv

import numpy as np 
from pyevtk.hl import imageToVTK




###############################################################################
# configurations
###############################################################################

# TODO: probably a better way to define these paths
if 'PVPYTHON' not in os.environ:
    os.environ['PVPYTHON'] = "/Applications/ParaView-5.11.0.app/Contents/bin/pvpython"

if 'TTK_PLUGIN' not in os.environ:
    os.environ['TTK_PLUGIN'] = "/Applications/ParaView-5.11.0.app/Contents/Plugins/TopologyToolKit.so"




###############################################################################
# helper functions
###############################################################################

def loss_landscape_to_vti(loss_landscape: List[List[float]], output_path: str) -> str:

    # check output folder
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # convert to an array and check loss_steps
    loss_landscape = np.array(loss_landscape)
    loss_steps = len(loss_landscape)

    # make sure we have a square matrix, convert if not
    if np.shape(loss_landscape)[-1] == 1:
        loss_steps = int(np.sqrt(loss_steps))
        loss_landscape = loss_landscape.reshape(loss_steps, loss_steps)

    # prepare the data to store in .vti files for ttk input
    loss_landscape_3d = loss_landscape.reshape(loss_steps, loss_steps, 1)

    # store the loss landscape results into binary files used for ttk
    imageToVTK(output_path, pointData={"Loss": loss_landscape_3d})

    # configure filename
    output_file = output_path + '.vti'

    return output_file



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
        {"x": points_0[i], "y0": points_1[i], "y1": points_2[i]}
        for i in range(len(nodeID))
    ]

    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')

    return persistence_barcode



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



def process_merge_tree_planar(input_file: str) -> dict:
    """ Process the CSV file produced by Paraview
    
    TODO: New representation includes the following

        "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Persistence","ClusterID","TreeID","isDummyNode","TrueNodeId","isImportantPair","isMultiPersPairNode","BranchNodeID","Points:0","Points:1","Points:2"
        0,1638,1599,3,483,38,1638,0,0,0,15,1,0,0,741.02,1638,0
        2,477.94,71,1,483,38,22.023,0,0,0,14,0,0,1,641.31,477.94,0
        2,477.94,71,1,483,38,22.023,0,0,1,14,0,0,1,741.02,477.94,0

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




###############################################################################
# main functions
###############################################################################

def compute_persistence_barcode(loss_landscape: List[List[float]], output_path: str) -> str:
    
    # convert loss_landscape into vti
    output_file_vti = loss_landscape_to_vti(loss_landscape, output_path)

    # compute persistence_barcode
    output_file_csv = compute_persistence_barcode_paraview(output_file_vti)

    # extract .csv and return persistence_barcode object
    persistence_barcode = process_persistence_barcode(output_file_csv)

    return persistence_barcode 



def compute_merge_tree(loss_landscape: List[List[float]], output_path: str) -> str:
    
    # convert loss_landscape into vti
    output_file_vti = loss_landscape_to_vti(loss_landscape, output_path)

    # compute merge_tree
    output_file_csv = compute_merge_tree_paraview(output_file_vti)

    # extract .csv and return merge_tree object
    merge_tree = process_merge_tree(output_file_csv)

    return merge_tree



def compute_merge_tree_planar(loss_landscape: List[List[float]], output_path: str) -> str:
    
    ### TODO: maybe just make planar=False an argument of compute_merge_tree

    # convert loss_landscape into vti
    output_file_vti = loss_landscape_to_vti(loss_landscape, output_path)

    # compute merge_tree
    output_file_csv = compute_merge_tree_planar_paraview(output_file_vti)

    # extract .csv and return merge_tree object
    merge_tree = process_merge_tree_planar(output_file_csv)

    return merge_tree



###############################################################################
# paraview functions
###############################################################################

def compute_persistence_barcode_paraview(input_file, output_folder=None):
    """ Run calculate_ttk_persistence_diagram.py using pvpython. """

    # configure output file name
    output_file = input_file.replace('.vti', '_PersistenceDiagram.csv')
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output 
    if not os.path.exists(output_file):

        # submit the command
        result = subprocess.run(
            [os.environ['PVPYTHON'], "calculate_ttk_persistence_diagram.py",
             "--ttk-plugin", os.environ['TTK_PLUGIN'],
             "--input-file", input_file,
             "--output-file", output_file],
            capture_output=True, text=True
        )
        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)

    return output_file



def compute_merge_tree_paraview(input_file, output_folder=None):
    """ Run calculate_ttk_merge_tree.py using pvpython. """

    # configure output file name
    output_file = input_file.replace('.vti', '_MergeTree.csv')
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output 
    if not os.path.exists(output_file):

        # submit the command
        result = subprocess.run(
            [os.environ['PVPYTHON'], "calculate_ttk_merge_tree.py",
             "--ttk-plugin", os.environ['TTK_PLUGIN'],
             "--input-file", input_file,
             "--output-file", output_file],
            capture_output=True, text=True
        )
        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)

    return output_file



def compute_merge_tree_planar_paraview(input_file, output_folder=None):
    """ Run calculate_ttk_merge_tree_planar.py using pvpython. """

    # configure output file name
    output_file = input_file.replace('.vti', '_MergeTreePlanar.csv')
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output 
    if not os.path.exists(output_file):

        # submit the command
        result = subprocess.run(
            [os.environ['PVPYTHON'], "calculate_ttk_merge_tree_planar.py",
             "--ttk-plugin", os.environ['TTK_PLUGIN'],
             "--input-file", input_file,
             "--output-file", output_file],
            capture_output=True, text=True
        )
        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)

    return output_file





