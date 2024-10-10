###############################################################################
# imports
###############################################################################

import os
import sys
from typing import List
import csv
import json

import subprocess
import numpy as np
from pyevtk.hl import imageToVTK, unstructuredGridToVTK
from pyevtk.vtk import VtkLine
from itertools import product

# code for graph constructon
import networkx as nx

from scipy.spatial import Delaunay
from pynndescent import NNDescent
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_dir + "/generate_loss_cubes/")
import torch

# from generate_loss_cubes.resnet import *
# from generate_loss_cubes.utils import *
# from generate_loss_cubes.net_pbc import *
# from generate_loss_cubes.functions import *
# from generate_loss_cubes.models.resnet_width import *
# from generate_loss_cubes.loss_landscapes import *
# from generate_loss_cubes.loss_landscapes.metrics import *

###############################################################################
# configurations
###############################################################################

# TODO: probably a better way to define these paths
if "PVPYTHON" not in os.environ:
    os.environ["PVPYTHON"] = "/Applications/ParaView-5.11.1.app/Contents/bin/pvpython"

if "TTK_PLUGIN" not in os.environ:
    os.environ["TTK_PLUGIN"] = (
        "/Applications/ParaView-5.11.1.app/Contents/Plugins/TopologyToolKit.so"
    )

###############################################################################
# device
###############################################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLAG = True if torch.cuda.is_available() else False

###############################################################################
# graph construction methods
###############################################################################


def compute_delaunay(loss_coords=None, return_graph=True, verbose=1):
    """Compute Delaunay triangulation and construct graph."""

    if verbose > 0:
        print(f"\n... Computing Delaunay triangulation")

    # compute Delaunay triangulation
    tri = Delaunay(loss_coords)

    # convert to Graph
    # TODO: this way maybe be very slow for larger graphs and may be over connected
    # TODO: see https://groups.google.com/g/networkx-discuss/c/D7fMmuzVBAw?pli=1
    # NOTE: fixed by adding each uniqure edge of each triangle
    G = nx.Graph()
    for path in tri.simplices:

        # define unique edges of the triangle
        e1 = {path[0], path[1]}
        e2 = {path[0], path[2]}
        e3 = {path[1], path[2]}

        # add edges to the graph
        G.add_edges_from([e1, e2, e3])

    # compute adjacency matrix
    A = nx.adjacency_matrix(G)

    # display some info
    if verbose > 0:
        print(f"    G.number_of_nodes() = {G.number_of_nodes()}")
        print(f"    G.number_of_edges() = {G.number_of_edges()}")
        print(f"    A.shape = {A.shape}")

    # return stuff
    if return_graph:
        return A, G
    return A


def compute_gabriel(loss_coords=None, return_graph=True, verbose=1):
    """Compute Gabriel graph."""

    if verbose > 0:
        print(f"\n... Computing Gabriel graph")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        from libpysal.weights import Gabriel

        # compute Gabriel graph
        gab = Gabriel(loss_coords)

    # convert to Graph
    G = nx.Graph()
    for node in range(len(loss_coords)):
        G.add_node(node)
        nbrs = gab.neighbors.get(node, [])
        for nbr in nbrs:
            G.add_edge(node, nbr)

    # compute adjacency matrix
    A = nx.adjacency_matrix(G)

    # display some info
    if verbose > 0:
        print(f"    G.number_of_nodes() = {G.number_of_nodes()}")
        print(f"    G.number_of_edges() = {G.number_of_edges()}")
        print(f"    A.shape = {A.shape}")

    # return stuff
    if return_graph:
        return A, G
    return A


def compute_aknn(
    loss_coords=None,
    n_neighbors=None,
    metric="euclidean",
    force_symmetric=False,
    return_graph=True,
    random_state=0,
    verbose=1,
):
    """Compute () Approximate kNN graph."""

    n_neighbors = n_neighbors or (4 * loss_coords.shape[1])

    if verbose > 0:
        print(
            f"\n... Computing Approximate k Nearest Neighbors graph (n_neighbors={n_neighbors}, force_symmetric={force_symmetric})"
        )

    # build (approximate) kNN index
    aknn = NNDescent(
        loss_coords,
        metric=metric,
        n_neighbors=n_neighbors,
        n_jobs=-1,
        random_state=random_state,
        verbose=False,
    )

    # get neighbors (not including self)
    nbrs = aknn.neighbor_graph[0][:, 1:]

    # construct (reciprocal) ANN Graph
    G = nx.Graph()

    if verbose > 1:
        print("    ", end="")

    for node in range(loss_coords.shape[0]):

        # display progress
        if (verbose > 1) and ((node % 1000) == 0):
            print(f"{node}.", end="")

        # add node to the graph
        G.add_node(node)

        # add edges between reciprocal neighbors
        for nbr in nbrs[node]:
            # require node to be a nbr of its nbrs?
            if force_symmetric and (node not in nbrs[nbr]):
                continue
            G.add_edge(node, nbr)

    if verbose > 1:
        print("")

    # compute adjacency matrix
    A = nx.adjacency_matrix(G)

    # display some info
    if verbose > 0:
        print(f"    G.number_of_nodes() = {G.number_of_nodes()}")
        print(f"    G.number_of_edges() = {G.number_of_edges()}")
        print(f"    A.shape = {A.shape}")

    # return stuff
    if return_graph:
        return A, G
    return A


def compute_rknn(
    loss_coords=None,
    n_neighbors=None,
    metric="euclidean",
    return_graph=True,
    random_state=0,
    verbose=1,
    n_jobs=1,
):
    """Compute () Approximate kNN graph."""

    n_neighbors = n_neighbors or (4 * loss_coords.shape[1])

    if verbose > 0:
        print(
            f"\n... Computing Reciprocal k Nearest Neighbors graph (n_neighbors={n_neighbors}, force_symmetric={force_symmetric})"
        )

    # build Reciprocal Adjacency matrix
    from reciprocal_isomap import ReciprocalIsomap

    r_isomap = ReciprocalIsomap(
        n_neighbors=n_neighbors,
        neighbors_mode="connectivity",
        metric=metric,
        n_jobs=n_jobs,
    )
    embedding = r_isomap.fit_transform(loss_coords)

    # compute adjacency matrix
    A = r_isomap.dist_matrix_.A == 1

    # construct (reciprocal) kNN Graph
    G = nx.Graph(A)

    # display some info
    if verbose > 0:
        print(f"    G.number_of_nodes() = {G.number_of_nodes()}")
        print(f"    G.number_of_edges() = {G.number_of_edges()}")
        print(f"    A.shape = {A.shape}")

    # return stuff
    if return_graph:
        return A, G
    return A


###############################################################################
# process ttk inputs
###############################################################################


def loss_landscape_to_vti(
    loss_landscape: List[List[float]] = None,
    loss_coords=None,
    loss_values=None,
    embedding=None,
    dim=2,
    output_path: str = "",
    loss_steps_dim1: int = None,
    loss_steps_dim2: int = None,
    loss_steps_dim3: int = None,
) -> str:

    # TODO: need to update this function to use loss_coords/values
    # TODO: should we deprecate loss_steps_dim1/dim2?

    # TODO: should we do this outside the function?
    output_path = output_path + "_ImageData"

    # check output folder
    output_folder = os.path.dirname(output_path)
    if len(output_folder) and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # convert to an array and check loss_steps
    loss_landscape = np.array(loss_landscape)

    # check if we have a square matrix
    if dim == 2:
        if loss_steps_dim1 == loss_steps_dim2:
            loss_steps = len(loss_landscape)
            # make sure we have a square matrix, convert if not
            if np.shape(loss_landscape)[-1] == 1:
                loss_steps = int(np.sqrt(loss_steps))
                loss_landscape = loss_landscape.reshape(loss_steps, loss_steps)
            # prepare the data to store in .vti files for ttk input
            loss_landscape_3d = loss_landscape.reshape(loss_steps, loss_steps, 1)
        else:
            # prepare the data to store in .vti files for ttk input
            loss_landscape_3d = loss_landscape.reshape(
                loss_steps_dim1, loss_steps_dim2, 1
            )
    elif dim == 3:
        if loss_steps_dim1 == loss_steps_dim2 == loss_steps_dim3:
            loss_steps = len(loss_landscape)
            if loss_steps_dim1 == int(np.cbrt(loss_steps)):
                loss_steps = int(np.cbrt(loss_steps))
            # make sure we have a square matrix, convert if not
            if np.shape(loss_landscape)[-1] == 1:
                loss_steps = int(np.cbrt(loss_steps))
                loss_landscape = loss_landscape.reshape(
                    loss_steps, loss_steps, loss_steps
                )
            # prepare the data to store in .vti files for ttk input
            loss_landscape_3d = loss_landscape.reshape(
                loss_steps, loss_steps, loss_steps
            )
        else:
            # prepare the data to store in .vti files for ttk input
            loss_landscape_3d = loss_landscape.reshape(
                loss_steps_dim1, loss_steps_dim2, loss_steps_dim3
            )

    # store the loss landscape results into binary files used for ttk
    imageToVTK(
        output_path,
        pointData={"Loss": loss_landscape_3d, "DataID": np.arange(loss_landscape.size)},
    )

    # configure filename
    output_file = output_path + ".vti"

    return output_file


def loss_landscape_to_vtu(
    loss_landscape: List[List[float]] = None,
    loss_coords=None,
    loss_values=None,
    embedding=None,
    output_path: str = "",
    graph_kwargs="aknn",
    n_neighbors=None,
) -> str:
    """

    TODO:
    - should we separate this into "loss_landscape_to_aknn", "aknn_to_vtu"
    - should we make graph_type an option "aknn", "gabriel", etc.
    - if the latter, maybe add graph_kwargs

    """

    # TODO: should we do this outside the function?
    output_path = output_path + "_UnstructuredGrid" + "_" + graph_kwargs
    output_path_nn = output_path + "_Neighbors" + "_" + graph_kwargs

    # check output folder
    output_folder = os.path.dirname(output_path)
    if len(output_folder) and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ### process landscape

    # convert to an array and check loss_steps
    loss_steps = None
    loss_dims = None
    if loss_coords is None:
        loss_landscape = np.array(loss_landscape)
        loss_steps = len(loss_landscape)

        # make sure we have a square matrix, convert if not
        # TODO: this assumes 2d, not sure we can figure out if higher d (?)
        if np.shape(loss_landscape)[-1] == 1:
            loss_steps = int(np.sqrt(loss_steps))
            loss_landscape = loss_landscape.reshape(loss_steps, loss_steps)

        # define loss coordinates
        loss_dims = np.ndim(loss_landscape)
        loss_coords = np.asarray(
            list(product(*[np.arange(loss_steps) for _ in range(loss_dims)]))
        )

    if loss_values is None:
        # TODO: extract loss values
        # TODO: will this match the coordinates (???)
        loss_values = loss_landscape.ravel()

    if loss_steps is None:
        loss_steps = np.ravel(loss_coords).max()

    if loss_dims is None:
        loss_dims = np.shape(loss_coords)[-1]

    ### construct graph

    # TODO: make these options more flexible
    if n_neighbors is None:
        n_neighbors = 4 * loss_dims

    # TODO: let user define the method
    # TODO: accept user defined kwargs
    # TODO: e.g., graph_kwargs=dict(kind="aknn", force_symmetric=True)
    if graph_kwargs == "aknn":
        A, G = compute_aknn(
            loss_coords=loss_coords,
            n_neighbors=n_neighbors,
            metric="euclidean",
            force_symmetric=False,
            return_graph=True,
            random_state=0,
            verbose=0,
        )
    elif graph_kwargs == "raknn":
        A, G = compute_aknn(
            loss_coords=loss_coords,
            n_neighbors=n_neighbors,
            metric="euclidean",
            force_symmetric=True,
            return_graph=True,
            random_state=0,
            verbose=0,
        )
    elif graph_kwargs == "rknn":
        A, G = compute_rknn(
            loss_coords=loss_coords,
            n_neighbors=n_neighbors,
            metric="euclidean",
            return_graph=True,
            n_jobs=-1,
            verbose=0,
        )
    elif graph_kwargs == "gabriel":
        A, G = compute_gabriel(loss_coords=loss_coords, return_graph=True, verbose=0)
    elif graph_kwargs == "delaunay":
        A, G = compute_delaunay(loss_coords=loss_coords, return_graph=True, verbose=0)
    else:
        print(f"Graph type {graph_kwargs} not recognized, using aknn")
        A, G = compute_aknn(
            loss_coords=loss_coords,
            n_neighbors=n_neighbors,
            metric="euclidean",
            force_symmetric=True,
            return_graph=True,
            random_state=0,
            verbose=0,
        )

    ### save neighbor information
    neighbors_dict = {}
    for n in G:
        neighbors_dict[str(n)] = str([_ for _ in G.neighbors(n)])
    # print(neighbors_dict)
    with open(f"{output_path_nn}.json", "w") as file:
        json.dump(neighbors_dict, file)

    ### process unstructured grid for VTK

    # TODO: should we use list or array?
    # extract the undirected edges as an array
    lines_unique = list(G.edges())

    # count the number of unique lines
    n_lines = len(lines_unique)

    # define points that belong in each line
    conn = np.ravel(lines_unique)

    # define offset of last vertex of each element
    offsets = (np.arange(n_lines) + 1) * 2

    # define array that defines the cell type of each element in the grid
    cell_types = np.repeat(VtkLine.tid, n_lines)

    # define dictionary with variables associated to each vertex.
    point_data = dict()
    point_data["Loss"] = loss_values.ravel()
    point_data["DataID"] = np.arange(len(loss_values))

    # define dictionary with variables associated to each cell.
    cell_data = None
    # cell_data = dict()
    # cell_data['Loss (mean)'] = np.mean(loss_values[lines_unique], axis=1).ravel()
    # cell_data['Loss (min)'] = np.min(loss_values[lines_unique], axis=1).ravel()
    # cell_data['Loss (max)'] = np.max(loss_values[lines_unique], axis=1).ravel()

    # TODO: compute PCA embedding of the coordinates (?)
    # TODO: accept a user defined embedding (?)
    if embedding is None:
        # embedding = PCA(3).fit_transform(loss_coords)
        embedding = PCA(n_components=2).fit_transform(loss_coords)

    # TODO: scale the loss and combine with PCA coordinates (?)
    # TODO: this assumes step size of 40 was used
    loss_values = loss_values.ravel()
    loss_values_scaled = MinMaxScaler((0, loss_steps)).fit_transform(
        loss_values[:, np.newaxis]
    )

    # TODO: scale embedding
    # from sklearn.preprocessing import MinMaxScaler
    # embedding = MinMaxScaler((0, loss_steps)).fit_transform(embedding)

    # combine first two PCs with scaled loss
    embedding = np.c_[embedding[:, :2], loss_values_scaled]

    # define x,y,z (ascontiguousarray to avoid VTK errors)
    x = np.ascontiguousarray(embedding[:, 0]).astype(float)
    y = np.ascontiguousarray(embedding[:, 1]).astype(float)
    z = np.ascontiguousarray(embedding[:, 2]).astype(float)

    # save as a VTK unstructured grid
    fn_loss_vtu = unstructuredGridToVTK(
        output_path,
        x=x,
        y=y,
        z=z,
        connectivity=conn,
        offsets=offsets,
        cell_types=cell_types,
        cellData=cell_data,
        pointData=point_data,
        fieldData=None,
    )

    # configure filename
    output_file = output_path + ".vtu"

    return output_file


###############################################################################
# process ttk outputs
###############################################################################


def process_persistence_barcode(input_file: str) -> list:
    """Process the CSV file produced by Paraview

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


def process_merge_tree_side(input_file: str) -> dict:
    """Process the CSV file produced by Paraview

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

    edge_file = input_file.replace(".csv", "_edge.csv")
    segmentation_file = input_file.replace(".csv", "_segmentation.csv")
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row["Points:0"]))
            pointsy.append(float(row["Points:1"]))
            pointsz.append(float(row["Points:2"]))

            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row["NodeId"]))

            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row["CriticalType"]))

            # find the start point of each branch
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

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if (
                this_x == pointsx[j]
                and this_y == pointsy[j]
                and this_z == pointsz[j]
                and i != j
            ):
                end[i] = 1
                end[j] = 1

    # convert to representation for the database
    nodes = [
        {
            "x": pointsx[i],
            "y": pointsy[i],
            "z": pointsz[i],
            "id": nodeID[i],
            "criticalType": branchID[i],
        }
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsy[i], pointsz[i]]

    edges = []

    with open(edge_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append(
                {
                    "sourceX": node_dict[int(row["downNodeId"])][0],
                    "sourceY": node_dict[int(row["downNodeId"])][1],
                    "targetX": node_dict[int(row["upNodeId"])][0],
                    "targetY": node_dict[int(row["upNodeId"])][1],
                }
            )

    segmentations = []
    with open(segmentation_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append(
                    {
                        "x": float(row["points:0"]),
                        "y": float(row["points:1"]),
                        "z": float(row["points:2"]),
                        "loss": float(row["Loss"]),
                    }
                )

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')

    return merge_tree


def process_merge_tree_3d(input_file: str) -> dict:
    """Process the CSV file produced by Paraview

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

    edge_file = input_file.replace(".csv", "_edge.csv")
    segmentation_file = input_file.replace(".csv", "_segmentation.csv")
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row["Points:0"]))
            pointsy.append(float(row["Points:1"]))
            pointsz.append(float(row["Points:2"]))

            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row["NodeId"]))

            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row["CriticalType"]))

            # find the start point of each branch
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

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if (
                this_x == pointsx[j]
                and this_y == pointsy[j]
                and this_z == pointsz[j]
                and i != j
            ):
                end[i] = 1
                end[j] = 1

    # convert to representation for the database
    nodes = [
        {
            "x": pointsx[i],
            "y": pointsy[i],
            "z": pointsz[i],
            "id": nodeID[i],
            "criticalType": branchID[i],
        }
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsx[i], pointsy[i], pointsz[i]]

    edges = []

    with open(edge_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append(
                {
                    "sourceX": node_dict[int(row["downNodeId"])][0],
                    "sourceY": node_dict[int(row["downNodeId"])][1],
                    "sourceZ": node_dict[int(row["downNodeId"])][2],
                    "targetX": node_dict[int(row["upNodeId"])][0],
                    "targetY": node_dict[int(row["upNodeId"])][1],
                    "targetZ": node_dict[int(row["upNodeId"])][2],
                }
            )

    segmentations = []
    with open(segmentation_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append(
                    {
                        "x": float(row["points:0"]),
                        "y": float(row["points:1"]),
                        "z": float(row["points:2"]),
                        "loss": float(row["Loss"]),
                    }
                )
            elif "Loss" in row:
                segmentations.append(
                    {
                        "loss": float(row["Loss"]),
                    }
                )

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')

    return merge_tree


def process_merge_tree_3d_vti(input_file: str) -> dict:
    """Process the CSV file produced by Paraview

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
    loss = []
    nodeID = []
    branchID = []
    start = []
    end = []

    # initialize some global variables
    root_x = 0

    edge_file = input_file.replace(".csv", "_edge.csv")
    segmentation_file = input_file.replace(".csv", "_segmentation.csv")
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row["Points:0"]))
            pointsy.append(float(row["Points:1"]))
            pointsz.append(float(row["Points:2"]))
            loss.append(float(row["Scalar"]))

            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row["NodeId"]))

            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row["CriticalType"]))

            # find the start point of each branch
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

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if (
                this_x == pointsx[j]
                and this_y == pointsy[j]
                and this_z == pointsz[j]
                and i != j
            ):
                end[i] = 1
                end[j] = 1

    # convert to representation for the database
    nodes = [
        {
            "x": pointsx[i],
            "y": pointsy[i],
            "z": pointsz[i],
            "id": nodeID[i],
            "criticalType": branchID[i],
        }
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsx[i], pointsy[i], pointsz[i], loss[i]]

    edges = []

    with open(edge_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append(
                {
                    "sourceX": node_dict[int(row["downNodeId"])][0],
                    "sourceY": node_dict[int(row["downNodeId"])][1],
                    "sourceZ": node_dict[int(row["downNodeId"])][3],
                    "targetX": node_dict[int(row["upNodeId"])][0],
                    "targetY": node_dict[int(row["upNodeId"])][1],
                    "targetZ": node_dict[int(row["upNodeId"])][3],
                }
            )

    segmentations = []
    with open(segmentation_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append(
                    {
                        "x": float(row["points:0"]),
                        "y": float(row["points:1"]),
                        "z": float(row["points:2"]),
                        "loss": float(row["Loss"]),
                    }
                )
            elif "Loss" in row:
                segmentations.append(
                    {
                        "loss": float(row["Loss"]),
                    }
                )

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')

    return merge_tree


def process_merge_tree_front(input_file: str) -> dict:
    """Process the CSV file produced by Paraview

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

    edge_file = input_file.replace(".csv", "_edge.csv")
    segmentation_file = input_file.replace(".csv", "_segmentation.csv")
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row["Points:0"]))
            pointsy.append(float(row["Points:1"]))
            pointsz.append(float(row["Points:2"]))

            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row["NodeId"]))

            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row["CriticalType"]))

            # find the start point of each branch
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

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if (
                this_x == pointsx[j]
                and this_y == pointsy[j]
                and this_z == pointsz[j]
                and i != j
            ):
                end[i] = 1
                end[j] = 1

    # convert to representation for the database
    nodes = [
        {
            "x": pointsx[i],
            "y": pointsy[i],
            "z": pointsz[i],
            "id": nodeID[i],
            "criticalType": branchID[i],
        }
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsx[i], pointsz[i]]

    edges = []

    with open(edge_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append(
                {
                    "sourceX": node_dict[int(row["downNodeId"])][0],
                    "sourceY": node_dict[int(row["downNodeId"])][1],
                    "targetX": node_dict[int(row["upNodeId"])][0],
                    "targetY": node_dict[int(row["upNodeId"])][1],
                }
            )

    segmentations = []
    with open(segmentation_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append(
                    {
                        "x": float(row["points:0"]),
                        "y": float(row["points:1"]),
                        "z": float(row["points:2"]),
                        "loss": float(row["Loss"]),
                    }
                )

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')

    return merge_tree


def process_merge_tree(input_file: str) -> dict:
    """Process the CSV file produced by Paraview

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
    persistences = []

    # initialize some global variables
    root_x = 0

    edge_file = input_file.replace(".csv", "_edge.csv")
    segmentation_file = input_file.replace(".csv", "_segmentation.csv")
    node_dict = {}

    # open the csv file and load the data into the lists
    with open(input_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pointsx.append(float(row["Points:0"]))
            pointsy.append(float(row["Points:1"]))
            pointsz.append(float(row["Points:2"]))

            # TODO: 'Point ID' doesn't exist in the current format
            # nodeID.append(int(row['Point ID']))
            # nodeID.append(int(row['VertexId']))
            nodeID.append(int(row["NodeId"]))

            # TODO: 'BranchNodeID' doesn't exist in the current format
            # branchID.append(int(row['BranchNodeID']))
            branchID.append(int(row["CriticalType"]))

            # find the start point of each branch
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

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if (
                this_x == pointsx[j]
                and this_y == pointsy[j]
                and this_z == pointsz[j]
                and i != j
            ):
                end[i] = 1
                end[j] = 1

    # convert to representation for the database
    nodes = [
        {"x": pointsx[i], "y": pointsy[i], "id": nodeID[i], "criticalType": branchID[i]}
        for i in range(len(start))
    ]

    for i in range(len(start)):
        node_dict[nodeID[i]] = [pointsx[i], pointsy[i]]

    edges = []

    with open(edge_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            edges.append(
                {
                    "sourceX": node_dict[int(row["downNodeId"])][0],
                    "sourceY": node_dict[int(row["downNodeId"])][1],
                    "targetX": node_dict[int(row["upNodeId"])][0],
                    "targetY": node_dict[int(row["upNodeId"])][1],
                }
            )

    segmentations = []
    with open(segmentation_file, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            # print(row)
            # TODO: 'Points:0' doesn't exist in the current vti format
            if "points:0" in row:
                segmentations.append(
                    {
                        "x": float(row["points:0"]),
                        "y": float(row["points:1"]),
                        "z": float(row["points:2"]),
                        "loss": float(row["Loss"]),
                        "SegmentationId": int(row["SegmentationId"]),
                    }
                )

    merge_tree = {"nodes": nodes, "edges": edges, "segmentations": segmentations}

    # TODO: save object using pickle?
    # configure output file name
    # output_file = input_file.replace('.csv', '_processed.pkl')

    return merge_tree


def process_merge_tree_planar(input_file: str) -> dict:
    """Process the CSV file produced by Paraview

    TODO: New representation includes the following

        "NodeId","Scalar","VertexId","CriticalType","RegionSize","RegionSpan","Persistence","ClusterID","TreeID","isDummyNode","TrueNodeId","isImportantPair","isMultiPersPairNode","BranchNodeID","Points:0","Points:1","Points:2"
        0,1638,1599,3,483,38,1638,0,0,0,15,1,0,0,741.02,1638,0
        2,477.94,71,1,483,38,22.023,0,0,0,14,0,0,1,641.31,477.94,0
        2,477.94,71,1,483,38,22.023,0,0,1,14,0,0,1,741.02,477.94,0

    """
    pointsx = []
    pointsy = []
    pointsz = []
    nodeID = []
    branchID = []
    start = []
    end = []
    persistences = []

    # initialize some global variables
    root_x = 0

    with open(input_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pointsx.append(float(row["Points:0"]))
            pointsy.append(float(row["Points:1"]))
            pointsz.append(float(row["Points:2"]))
            nodeID.append(int(row["NodeId"]))
            branchID.append(int(row["BranchNodeID"]))
            persistences.append(float(row["Persistence"]))

            # find the start point of each branch
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

    # find the end point of each branch
    for i in range(len(start)):
        this_x = pointsx[i]
        this_y = pointsy[i]
        this_z = pointsz[i]
        for j in range(len(start)):
            if (
                this_x == pointsx[j]
                and this_y == pointsy[j]
                and this_z == pointsz[j]
                and i != j
            ):
                end[i] = 1
                end[j] = 1

    # verify that the start and end points are correct
    temp_structure = []
    for i in range(len(start)):
        t = {
            "start": start[i],
            "end": end[i],
            "x": pointsx[i],
            "y": pointsy[i],
            "z": pointsz[i],
            "nodeID": nodeID[i],
            "branchID": branchID[i],
            "Persistence": persistences[i],
        }
        temp_structure.append(t)

    nodes = []
    for item in temp_structure:
        nodes.append(
            {
                "id": item["nodeID"],
                "x": item["x"],
                "y": item["y"],
                "Persistence": item["Persistence"],
            }
        )

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

    res = {
        "nodes": nodes,
        "edges": edges,
    }

    # print(res)
    return res


###############################################################################
# main functions
###############################################################################


def compute_persistence_barcode(
    loss_landscape: List[List[float]] = None,
    loss_coords: List[List[float]] = None,
    loss_values: List[float] = None,
    embedding: List[List[float]] = None,
    dim=2,
    loss_steps_dim1: int = None,
    loss_steps_dim2: int = None,
    loss_steps_dim3: int = None,
    output_path: str = "",
    vtk_format: str = "vti",
    graph_kwargs: str = "aknn",
    n_neighbors=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> str:

    # convert loss_landscape into a vtk format
    output_file_vtk = None
    if vtk_format.lower() == "vti":

        # convert loss_landscape into a (.vti) image data format
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            dim=dim,
            loss_steps_dim1=loss_steps_dim1,
            loss_steps_dim2=loss_steps_dim2,
            loss_steps_dim3=loss_steps_dim3,
            output_path=output_path,
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            output_path=output_path,
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors,
        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute persistence_barcode
    output_file_csv = compute_persistence_barcode_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold,
        threshold_is_absolute=threshold_is_absolute,
    )

    # extract .csv and return persistence_barcode object
    persistence_barcode = process_persistence_barcode(output_file_csv)

    return persistence_barcode


def compute_merge_tree(
    loss_landscape: List[List[float]] = None,
    loss_coords: List[List[float]] = None,
    loss_values: List[float] = None,
    embedding: List[List[float]] = None,
    dim=2,
    loss_steps_dim1: int = None,
    loss_steps_dim2: int = None,
    loss_steps_dim3: int = None,
    output_path: str = "",
    vtk_format: str = "vti",
    graph_kwargs: str = "aknn",
    n_neighbors=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> str:

    # convert loss_landscape into a vtk format
    output_file_vtk = None
    if vtk_format.lower() == "vti":

        # convert loss_landscape into a (.vti) image data format
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            dim=dim,
            loss_steps_dim1=loss_steps_dim1,
            loss_steps_dim2=loss_steps_dim2,
            loss_steps_dim3=loss_steps_dim3,
            output_path=output_path,
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            output_path=output_path,
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors,
        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute merge_tree
    output_file_csv = compute_merge_tree_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold,
        threshold_is_absolute=threshold_is_absolute,
    )

    # extract .csv and return merge_tree object
    merge_tree = process_merge_tree(output_file_csv)

    return merge_tree


def compute_merge_tree_planar(
    loss_landscape: List[List[float]] = None,
    loss_coords: List[List[float]] = None,
    loss_values: List[float] = None,
    embedding: List[List[float]] = None,
    dim=2,
    loss_steps_dim1: int = None,
    loss_steps_dim2: int = None,
    loss_steps_dim3: int = None,
    output_path: str = "",
    vtk_format: str = "vti",
    graph_kwargs: str = "aknn",
    n_neighbors=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> str:

    ### TODO: maybe just make planar=False an argument of compute_merge_tree

    # convert loss_landscape into a vtk format
    output_file_vtk = None
    if vtk_format.lower() == "vti":

        # convert loss_landscape into a (.vti) image data format
        output_file_vtk = loss_landscape_to_vti(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            dim=dim,
            loss_steps_dim1=loss_steps_dim1,
            loss_steps_dim2=loss_steps_dim2,
            loss_steps_dim3=loss_steps_dim3,
            output_path=output_path,
        )

    elif vtk_format.lower() == "vtu":

        # convert loss_landscape into a (.vtu) unstructured grid format
        output_file_vtk = loss_landscape_to_vtu(
            loss_landscape=loss_landscape,
            loss_coords=loss_coords,
            loss_values=loss_values,
            embedding=embedding,
            output_path=output_path,
            graph_kwargs=graph_kwargs,
            n_neighbors=n_neighbors,
        )

    else:
        raise ValueError("VTK format not recognized, please specify vti or vtu")

    # compute merge_tree
    output_file_csv = compute_merge_tree_planar_paraview(
        output_file_vtk,
        persistence_threshold=persistence_threshold,
        threshold_is_absolute=threshold_is_absolute,
    )

    # extract .csv and return merge_tree object
    merge_tree = process_merge_tree_planar(output_file_csv)

    return merge_tree


def load_pinn_model(file_path, verbose=0):
    """Given a file name, compute the error of the model."""

    # get model file name
    model_file_name = file_path.split("_pinn_")[1].split("_dim3")[0] + ".pt"
    model_source = file_path.split("source")[1].split("_")[0]
    model_seed = file_path.split("seed")[1].split("_")[0]
    model_lr = file_path.split("lr")[1].split("_")[0]
    model_beta = file_path.split("beta")[1].split("_")[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_file_name = (
        model_file_name.split("_lr")[0]
        + "_source"
        + model_source
        + "_seed"
        + model_seed
        + ".pt"
    )
    folder_name = f"/generate_loss_cubes/saved_models/PINN_checkpoints/PINN_convection/lr_{model_lr}/beta_{model_beta}/"
    model_file_path = parent_dir + folder_name + model_file_name

    # return NaN if file not found
    if not os.path.exists(model_file_path):
        print("model file does not exist")
        if verbose > 0:
            print(f"[!] {model_file_path} does not exist!!!")
        return np.NaN

    # load model
    model = torch.load(model_file_path, map_location=device)
    # print("load model successfully")
    model.dnn.eval()

    return model


##############################################################################
# paraview functions
###############################################################################


def compute_persistence_barcode_paraview(
    input_file,
    output_folder=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> str:
    """Run calculate_ttk_persistence_diagram.py using pvpython."""

    # configure simplification str (to avoid recomputing)
    # TODO: a bit long, maybe shorten in the future, e.g.,
    # - current : "_PersistenceThreshold_0.0_ThresholdIsAbsolute_0_"
    # - option 1: "_simplify_0.0_" vs. "_simplify_0.0_absolute_"
    # - option 2: "_simplify_0.0_absolute_0_"
    simplification_str = f"_PersistenceThreshold_{persistence_threshold}_ThresholdIsAbsolute_{int(threshold_is_absolute)}"

    # configure output file name
    output_file = (
        input_file.split(".vt")[0] + simplification_str + "_PersistenceDiagram.csv"
    )
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output
    if not os.path.exists(output_file):

        # format the command
        _command = [
            os.environ["PVPYTHON"],
            f"{os.path.dirname(__file__)}/calculate_ttk_persistence_diagram.py",
            f"--ttk-plugin={os.environ['TTK_PLUGIN']}",
            f"--input-file={input_file}",
            f"--output-file={output_file}",
            f"--persistence-threshold={persistence_threshold}",
            f"--threshold-is-absolute" if threshold_is_absolute else "",
        ]
        _command = list(filter(None, _command))
        # print(" ".join(_command))

        # submit the command
        result = subprocess.run(_command, capture_output=True, text=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

    return output_file


def compute_merge_tree_paraview(
    input_file,
    output_folder=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> str:
    """Run calculate_ttk_merge_tree.py using pvpython."""

    # configure simplification str (to avoid recomputing)
    # TODO: a bit long, maybe shorten in the future, e.g.,
    # - current : "_PersistenceThreshold_0.0_ThresholdIsAbsolute_0_"
    # - option 1: "_simplify_0.0_" vs. "_simplify_0.0_absolute_"
    # - option 2: "_simplify_0.0_absolute_0_"
    simplification_str = f"_PersistenceThreshold_{persistence_threshold}_ThresholdIsAbsolute_{int(threshold_is_absolute)}"

    # configure output file name
    output_file = input_file.split(".vt")[0] + simplification_str + "_MergeTree.csv"
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output
    if not os.path.exists(output_file):

        # format the command
        _command = [
            os.environ["PVPYTHON"],
            f"{os.path.dirname(__file__)}/calculate_ttk_merge_tree.py",
            f"--ttk-plugin={os.environ['TTK_PLUGIN']}",
            f"--input-file={input_file}",
            f"--output-file={output_file}",
            f"--persistence-threshold={persistence_threshold}",
            f"--threshold-is-absolute" if threshold_is_absolute else "",
        ]
        _command = list(filter(None, _command))
        # print("_".join(_command))

        # submit the command
        result = subprocess.run(_command, capture_output=True, text=True)
        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)

    return output_file


def compute_merge_tree_planar_paraview(
    input_file,
    output_folder=None,
    persistence_threshold: float = 0.0,
    threshold_is_absolute: bool = False,
) -> str:
    """Run calculate_ttk_merge_tree_planar.py using pvpython."""

    # configure simplification str (to avoid recomputing)
    # TODO: a bit long, maybe shorten in the future, e.g.,
    # - current : "_PersistenceThreshold_0.0_ThresholdIsAbsolute_0_"
    # - option 1: "_simplify_0.0_" vs. "_simplify_0.0_absolute_"
    # - option 2: "_simplify_0.0_absolute_0_"
    simplification_str = f"_PersistenceThreshold_{persistence_threshold}_ThresholdIsAbsolute_{int(threshold_is_absolute)}"

    # configure output file name
    output_file = (
        input_file.split(".vt")[0] + simplification_str + "_MergeTreePlanar.csv"
    )
    if output_folder is not None:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    # check for existing output
    if not os.path.exists(output_file):

        # format the command
        _command = [
            os.environ["PVPYTHON"],
            f"{os.path.dirname(__file__)}/calculate_ttk_merge_tree_planar.py",
            f"--ttk-plugin={os.environ['TTK_PLUGIN']}",
            f"--input-file={input_file}",
            f"--output-file={output_file}",
            f"--persistence-threshold={persistence_threshold}",
            f"--threshold-is-absolute" if threshold_is_absolute else "",
        ]
        _command = list(filter(None, _command))
        # print("_".join(_command))

        # submit the command
        result = subprocess.run(_command, capture_output=True, text=True)
        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)

    return output_file
