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
import operation.torch_cka.cka as torch_cka

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
args = parser.parse_args()

# Model Name
model_name = args.model
# set the subdataset list
subdataset_list = ['original','all']

# define loss function
criterion = torch.nn.CrossEntropyLoss()

# prepare all the models
model_list = []
if model_name == 'VIT':
    model_list = operation.get_vit_model_list(dataset_name, subdataset_list, model_name)
elif model_name == 'RESNET18':
    model_list = operation.get_torchvision_model_list(subdataset_list, model_name, num_classes)

# prepare the test dataloader
cifar10_testing = []
x_augmented, y_augmented = load_cifar10c(n_examples=int(CIFAR10_SIZE*CIFAR10C_PERCENTAGE), data_dir='../data/')
for i in range(len(x_augmented)):
    cifar10_testing.append([x_augmented[i].float().reshape(3, 32, 32), y_augmented[i].long()])
cifar10_original = torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_loader  = torch.utils.data.DataLoader(cifar10_original, batch_size=1,drop_last=False)
cifar10_original_iterator = iter(cifar10_original_loader)
for i in range(int(len(cifar10_original_loader)*CIFAR10C_PERCENTAGE)):
    x_original, y_original = cifar10_original_iterator.__next__()
    x_original = torch.tensor(np.array(x_original.reshape(3, 32, 32)))
    cifar10_testing.append([x_original, y_original])
cifar10_original_test = torchvision.datasets.CIFAR10('../data/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_test_loader  = torch.utils.data.DataLoader(cifar10_original_test, batch_size=1,drop_last=False)
cifar10_original_test_iterator = iter(cifar10_original_test_loader)
for i in range(int(len(cifar10_original_test_loader)*CIFAR10C_PERCENTAGE)):
    x_original, y_original = cifar10_original_test_iterator.__next__()
    x_original = torch.tensor(np.array(x_original.reshape(3, 32, 32)))
    cifar10_testing.append([x_original, y_original])

# define testing data loader
dataloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=10, shuffle=False)

# calculate the CKA distance for models using torch CKA
cka = torch_cka.CKA(model_list[0], model_list[1], device="cpu")
cka.compare(dataloader)
results = cka.export()
print(results)