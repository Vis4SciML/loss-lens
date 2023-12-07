# libraries
import numpy as np
import copy
import argparse
import torch
import torch.nn
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
import torch
import torchvision
from utils import *
from robustbench.data import load_cifar10c
import numpy.ma as ma

# from pyevtk.hl import gridToVTK
from pyevtk.hl import imageToVTK

from operation import functions as operation

# training hyperparameters
batch_size_train = 64
batch_size_test = 128
num_classes = 10
CIFAR10_SIZE = 60000
CIFAR10C_PERCENTAGE = 0.01
dataset_name = 'CIFAR10'

# Hessian perb parameters
STEPS = 25
START = -0.5
END = 0.5

# Prediction distribution parameters
MAX_NUM = 100
INPUT_DATA_PERCENTAGE = 0.01

def get_params(model_orig, model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='VIT', help='trained model for testing')
parser.add_argument('--reshapex', default=464, help='reshape x for cka distance calculation')
parser.add_argument('--reshapey', default=128, help='reshape y for cka distance calculation')
parser.add_argument('--losssteps', default=4, help='steps for loss landscape calculation')
parser.add_argument('--device', default="local", help='device to do the back-end calculation')
args = parser.parse_args()

# Loss Landscape parameters
LOSS_STEPS = int(args.losssteps)

# Model Name
model_name = args.model

# set the subdataset list
# subdataset_list = ['original', 'threshold_{20}', 'threshold_{40}', 'threshold_{60}', 'threshold_{80}', 'all']
subdataset_list = ['original', 'threshold_{80}']

# set the client and database name
client = "mongodb://localhost:27017/"
database_name = "mydatabase"
if args.device == "nersc":
    client = "mongodb07.nersc.gov"
    database_name = "losslensdb"

# set all the collection names
collection_name_model_euclidean_distance_similarity = "model_euclidean_distance_similarity"
collection_name_cka_model_similarity = "cka_model_similarity"
collection_name_layer_euclidean_distance_similarity = "layer_euclidean_distance_similarity"
collection_name_hessian_contour = "hessian_contour"
collection_name_hessian_contour_model_information = "hessian_contour_model_information"
collection_name_loss_landscapes_contour = "loss_landscapes_contour"
collection_name_hessian_loss_landscape = "hessian_loss_landscape"
collection_name_loss_landscapes_contour_3d = "loss_landscapes_contour_3d"
collection_name_loss_landscapes_global = "loss_landscapes_global"
collection_name_global_structure = "global_structure"
collection_name_loss_landscapes_detailed_similarity = "loss_landscapes_detailed_similarity"
collection_name_model_analysis_information = "model_analysis_information"
collection_name_prediction_distribution = "prediction_distribution"
collection_name_training_loss_landscape = "training_loss_landscape"
collection_name_loss_landscapes_contour_3d_training = "loss_landscapes_contour_3d_training"
collection_name_layer_cka_similarity = "layer_cka_similarity"
collection_name_layer_torch_cka_similarity = "layer_torch_cka_similarity"

# define loss function
criterion = torch.nn.CrossEntropyLoss()

# generate training 3D loss landscapes for random projection
print("Generating training 3D loss landscapes for random projection...")
loss_data_fin_list_train_3d = operation.calculate_vit_training_3d_loss_landscapes_random_projection(dataset_name, subdataset_list, model_name, criterion, LOSS_STEPS)
loss_data_fin_array_train_3d = np.array(loss_data_fin_list_train_3d)
loss_data_fin_array_train_3d = np.where(np.isnan(loss_data_fin_array_train_3d), ma.array(loss_data_fin_array_train_3d, mask=np.isnan(loss_data_fin_array_train_3d)).mean(axis=0), loss_data_fin_array_train_3d)
loss_data_fin_array_train_3d = np.where(np.isposinf(loss_data_fin_array_train_3d), ma.array(loss_data_fin_array_train_3d, mask=np.isposinf(loss_data_fin_array_train_3d)).max(axis=0), loss_data_fin_array_train_3d)
loss_data_fin_array_train_3d = np.where(np.isneginf(loss_data_fin_array_train_3d), ma.array(loss_data_fin_array_train_3d, mask=np.isneginf(loss_data_fin_array_train_3d)).min(axis=0), loss_data_fin_array_train_3d)
loss_data_fin_list_train_3d = loss_data_fin_array_train_3d.tolist()
print("Finished generating training 3D loss landscapes for random projection.")

# # save the results to the database
# for i in tqdm(range(len(loss_data_fin_list_train_3d)), desc="Working on 3D loss landscapes for random projection"):
#     record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS, "loss_data_fin_3d": loss_data_fin_list_train_3d[i]}
#     record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_contour_3d_training, record)
#     print("One record of training 3D loss landscapes for random projection is stored in the database with id: " + str(record_id))

# prepare the data to store in .vti files for ttk input
nx, ny, nz = LOSS_STEPS - 1, LOSS_STEPS - 1, LOSS_STEPS - 1
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Variables
pressure = np.zeros(ncells).reshape((nx, ny, nz), order='C')

# store the loss landscape results into binary files used for ttk
for i in range(len(loss_data_fin_list_train_3d)):
    FILE_PATH = '../ttk/input_binary_for_ttk/CIFAR10_' + model_name + '_' + subdataset_list[i] + '_loss_cifar10_training_3d_contour'
    imageToVTK(FILE_PATH, cellData={"Cell": pressure}, pointData={"Loss": np.array(loss_data_fin_list_train_3d[i])})
