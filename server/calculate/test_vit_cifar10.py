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
CIFAR10C_PERCENTAGE = 0.001
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

# define loss function
criterion = torch.nn.CrossEntropyLoss()

# calculate the torch CKA similarity between layers
print("Calculating the CKA similarity between model layers...")
cka_result_all = operation.calculate_vit_layer_torch_cka_similarity(dataset_name, subdataset_list, model_name)
print("Finished calculating the CKA similarity between model layers.")

# print the torch CKA similarity between models
for i in range(len(cka_result_all)):
    cka_result = cka_result_all[i]
    for j in range(len(cka_result)):
        print("Torch CKA similarity between model " + str(i) + " and model " + str(j) + " is: ", cka_result[j])
