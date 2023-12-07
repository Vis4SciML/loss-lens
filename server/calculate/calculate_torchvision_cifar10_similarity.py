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

# calculate the euclidean distance and global structure between models
print("Calculating the euclidean distance between models...")
X_transformed, global_structure = operation.calculate_torchvision_model_similarity_global_structure(subdataset_list,model_name,num_classes)
print("Finished calculating the euclidean distance between models.")

# print the euclidean distance matrix and the global structure
print("Euclidean distance between models: ", X_transformed)
print("Euclidean distance matrix shape: ", X_transformed.shape)

# save the model euclidean distance similarity results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'modelMDS': X_transformed.tolist()}
record_id = operation.store_into_database(client, database_name, collection_name_model_euclidean_distance_similarity,record)
print("One record of model_euclidean_distance_similarity has been successfully inserted with ID " + str(record_id))

# calculate the cka distance between models
print("Calculating the CKA distance between models...")
reshape_x = int(args.reshapex)
reshape_y = int(args.reshapey)
linear_cka_embedding, kernel_cka_embedding, linear_cka_figure, kernel_cka_figure = operation.calculate_torchvision_model_cka_similarity_global_structure(subdataset_list, model_name, num_classes, reshape_x, reshape_y)
print("Finished calculating the CKA distance between models.")

# print the CKA distances matrix and the MDS results
print("Linear CKA similarity distance between models: ", linear_cka_embedding)
print("Linear CKA similarity matrix shape: ", linear_cka_embedding.shape)
print("RBF kernel CKA similarity distance between models: ", kernel_cka_embedding)
print("RBF kernel CKA similarity matrix shape: ", kernel_cka_embedding.shape)

# save the model CKA similarity results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'linearCKA_MDS': linear_cka_embedding.tolist(),'kernelCKA_MDS': kernel_cka_embedding.tolist()}
record_id = operation.store_into_database(client, database_name, collection_name_cka_model_similarity, record)
print("One record of cka_model_similarity has been successfully inserted with ID " + str(record_id))
print("CKA model similarity information have been saved into MongoDB.")

# print the global structure
print("Global structure of all the models based on Euclidean distance: ", global_structure)
print("Global structure of all the models based on Linear CKA similarity: ", linear_cka_figure)
print("Global structure of all the models based on RBF kernel CKA similarity: ", kernel_cka_figure)

# save the model global structure results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'global_structure': global_structure, 'linearCKA_global_structure': linear_cka_figure, 'kernelCKA_global_structure': kernel_cka_figure}
record_id = operation.store_into_database(client, database_name, collection_name_global_structure, record)
print("One record of global_structure has been successfully inserted with ID " + str(record_id))
print("Model global structure information have been saved into MongoDB.")
