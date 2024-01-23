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
parser.add_argument('--model', default='RESNET18', help='trained model for testing')
parser.add_argument('--reshapex', default=228, help='reshape x for cka distance calculation')
parser.add_argument('--reshapey', default=64, help='reshape y for cka distance calculation')
parser.add_argument('--losssteps', default=4, help='steps for loss landscape calculation')
parser.add_argument('--device', default="local", help='device to do the back-end calculation')
parser.add_argument('--number_of_workers', default=0, help='number of workers')
args = parser.parse_args()

# workers setting
number_of_workers = int(args.number_of_workers)

# Loss Landscape parameters
LOSS_STEPS = int(args.losssteps)

# Model Name
model_name = args.model

# set the subdataset list
subdataset_list = ['original', 'threshold_{20}', 'threshold_{40}', 'threshold_{60}', 'threshold_{80}', 'all']

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

# calculate the torch CKA similarity between layers
print("Calculating the CKA similarity between model layers...")
cka_result_all = operation.calculate_torchvision_layer_torch_cka_similarity(subdataset_list, model_name, num_classes)
print("Finished calculating the CKA similarity between model layers.")

# save the torch CKA similarity into database
for i in range(len(cka_result_all)):
    cka_result = cka_result_all[i]
    for j in range(len(cka_result)):
        record = {'dataset': dataset_name, 'model_type': model_name, 'model1': str(i), 'model2': str(j), 'torchCKA_similarity': cka_result[j]["CKA"].tolist()}
        record_id = operation.store_into_database(client, database_name, collection_name_layer_torch_cka_similarity, record)
        print("One record of layer_torch_cka_similarity has been successfully inserted with ID " + str(record_id))
print("Layer torch CKA similarity information have been saved into MongoDB.")
