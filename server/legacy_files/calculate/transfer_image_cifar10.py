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
import numpy.ma as ma
from matplotlib import pyplot as plt

# cifar10c library
from robustbench.data import load_cifar10c

from operation import functions as operation

# hyperparameters
CIFAR10_SIZE = 60000
CIFAR10C_PERCENTAGE = 0.01

# dataset name
dataset_name = 'CIFAR10'

# set the client and database name
client = "mongodb://localhost:27017/"
database_name = "mydatabase"
# set all the collection names
collection_name_labels = "prediction_distribution_labels"

# prepare the testing dataset for computation
cifar10_prediction_distribution = []
x_augmented, y_augmented = load_cifar10c(n_examples=int(CIFAR10_SIZE*CIFAR10C_PERCENTAGE), data_dir='../data/')
for i in range(len(x_augmented)):
    cifar10_prediction_distribution.append([x_augmented[i].float(), y_augmented[i].long()])
cifar10_original = torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_loader  = torch.utils.data.DataLoader(cifar10_original, batch_size=1,drop_last=False)
cifar10_original_iterator = iter(cifar10_original_loader)
for i in range(int(len(cifar10_original_loader)*CIFAR10C_PERCENTAGE)):
    x_original, y_original = cifar10_original_iterator.__next__()
    cifar10_prediction_distribution.append([x_original, y_original])
cifar10_original_test = torchvision.datasets.CIFAR10('../data/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_test_loader  = torch.utils.data.DataLoader(cifar10_original_test, batch_size=1,drop_last=False)
cifar10_original_test_iterator = iter(cifar10_original_test_loader)
for i in range(int(len(cifar10_original_test_loader)*CIFAR10C_PERCENTAGE)):
    x_original, y_original = cifar10_original_test_iterator.__next__()
    cifar10_prediction_distribution.append([x_original, y_original])

# define testing data loader for prediction distribution calculation
distribution_test_loader = torch.utils.data.DataLoader(cifar10_prediction_distribution, batch_size=1, shuffle=False)

# set iterator to test_loader
test_loader_iter = iter(distribution_test_loader)

# transfer and save images and labels
cifar10_prediction_distribution_labels = []

# save the images in local directory
for i in range(len(distribution_test_loader)):
    x, y = test_loader_iter.__next__()
    cifar10_prediction_distribution_labels.append(y.item())
    x = x.reshape((3, 32, 32))
    # input is 3x32x32. imshow needs 32x32x3
    x = x.permute(1, 2, 0)
    plt.clf()
    plt.axis('off')
    plt.imshow(x)
    plt.savefig("CIFAR10_figures/figure_" + str(i) + ".png", bbox_inches='tight')
    print("saved " + str(i+1) + " image")

# save the labels in the database
# save the labels into database
record = {"dataset": dataset_name, "labels": cifar10_prediction_distribution_labels}
record_id = operation.store_into_database(client, database_name, collection_name_labels, record)
print("One record of prediction_distribution_labels has been successfully inserted with ID " + str(record_id))