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
CIFAR10C_PERCENTAGE = 1
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

# prepare the testing dataset for prediction distribution computation
print("Loading CIFAR10 and CIFAR10-C dataset for prediction distribution...")
cifar10_prediction_distribution = []
x_augmented, y_augmented = load_cifar10c(n_examples=int(CIFAR10_SIZE * INPUT_DATA_PERCENTAGE), data_dir='../data/')
for i in range(len(x_augmented)):
    cifar10_prediction_distribution.append([x_augmented[i].float(), y_augmented[i].long()])
cifar10_original = torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_loader = torch.utils.data.DataLoader(cifar10_original, batch_size=1, drop_last=False, num_workers=number_of_workers)
cifar10_original_iterator = iter(cifar10_original_loader)
for i in range(int(len(cifar10_original_loader) * INPUT_DATA_PERCENTAGE)):
    x_original, y_original = cifar10_original_iterator.__next__()
    cifar10_prediction_distribution.append([x_original, y_original])
cifar10_original_test = torchvision.datasets.CIFAR10('../data/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_test_loader = torch.utils.data.DataLoader(cifar10_original_test, batch_size=1, drop_last=False, num_workers=number_of_workers)
cifar10_original_test_iterator = iter(cifar10_original_test_loader)
for i in range(int(len(cifar10_original_test_loader) * INPUT_DATA_PERCENTAGE)):
    x_original, y_original = cifar10_original_test_iterator.__next__()
    cifar10_prediction_distribution.append([x_original, y_original])  
# define testing data loader for prediction distribution calculation
distribution_test_loader = torch.utils.data.DataLoader(cifar10_prediction_distribution, batch_size=1, shuffle=False, num_workers=number_of_workers)
print("Finish loading the testing data for prediction distribution calculation.")

# get the models
print("Getting the models for prediction distribution computation...")
# subdataset_list = ['original', 'threshold_{20}', 'threshold_{40}', 'threshold_{60}', 'threshold_{80}', 'all']
model_list = operation.get_torchvision_model_list(subdataset_list, model_name, num_classes)
print("Finish getting the models for prediction distribution computation.")

# calculate the prediction distribution of each pair of models
print("Calculating the prediction distribution of all models...")
for k in range(5, 15, 5):
    for i in range(len(model_list)):
        for j in range(len(model_list)):
            # select two models
            model_one = model_list[i]
            model_two = model_list[j]
            # calculate the prediction distribution
            correct_correct, correct_wrong, wrong_correct, wrong_wrong = operation.calculate_torchvision_model_prediction_distribution(model_one, model_two, distribution_test_loader, MAX_NUM)
            print("Finish calculating the prediction distribution of one pair of models.")
            print("Correct-Correct: " + str(correct_correct) + ", Correct-Wrong: " + str(correct_wrong) + ", Wrong-Correct: " + str(wrong_correct) + ", Wrong-Wrong: " + str(wrong_wrong))
            # store the results into the database
            record = {"dataset": dataset_name, 'model_type': model_name, "model1": subdataset_list[i], "model2": subdataset_list[j], "correct_correct": correct_correct, "correct_wrong": correct_wrong, "wrong_correct": wrong_correct, "wrong_wrong": wrong_wrong, "threshold": k}
            record_id = operation.store_into_database(client, database_name, collection_name_prediction_distribution, record)
            print("One record of prediction_distribution has been successfully inserted with ID " + str(record_id))
print("Finish calculating the prediction distribution of all models.")
