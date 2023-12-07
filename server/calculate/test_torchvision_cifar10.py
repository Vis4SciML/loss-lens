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
import matplotlib.pyplot as plt

# from pyevtk.hl import gridToVTK
from pyevtk.hl import imageToVTK

from operation import functions as operation
import operation.loss_landscapes as loss_landscapes
import operation.loss_landscapes.metrics as metrics
import operation.loss_landscapes.metrics.sl_metrics as sl_metrics
import operation.loss_landscapes.main as loss_landscapes_main
import operation.torch_cka.cka as torch_cka

# training hyperparameters
batch_size_train = 64
batch_size_test = 128
num_classes = 10
CIFAR10_SIZE = 60000
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
parser.add_argument('--losssteps', default=30, help='steps for loss landscape calculation')
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
subdataset_list = ['original']

# define loss function
criterion = torch.nn.CrossEntropyLoss()

# # calculate the training loss landscape
# print("Calculating the training loss landscape...")
# loss_data_fin_list_train, model_info_list_train, max_loss_value_list_train, min_loss_value_list_train = operation.calculate_torchvision_model_training_loss_landscapes(subdataset_list, model_name, num_classes, criterion,LOSS_STEPS)
# print("Finished calculating the training loss landscape.")

# print(loss_data_fin_list_train[0])

# # plot the training loss landscape
# plt.contour(loss_data_fin_list_train[0], levels=30)
# plt.title('Loss Contours around Trained Model Original')
# # save plot to file
# plt.savefig('loss_mnist_2d_contour_org.png')

# loss_data_fin_array_train = np.array(loss_data_fin_list_train)
# loss_data_fin_array_train = np.where(np.isnan(loss_data_fin_array_train), ma.array(loss_data_fin_array_train, mask=np.isnan(loss_data_fin_array_train)).mean(axis=0), loss_data_fin_array_train)
# loss_data_fin_array_train = np.where(np.isposinf(loss_data_fin_array_train), ma.array(loss_data_fin_array_train, mask=np.isposinf(loss_data_fin_array_train)).max(axis=0), loss_data_fin_array_train)
# loss_data_fin_array_train = np.where(np.isneginf(loss_data_fin_array_train), ma.array(loss_data_fin_array_train, mask=np.isneginf(loss_data_fin_array_train)).min(axis=0), loss_data_fin_array_train)
# loss_data_fin_list_train = loss_data_fin_array_train.tolist()

# print(loss_data_fin_list_train[0])

# # plot the training loss landscape
# plt.clf()
# plt.contour(loss_data_fin_list_train[0], levels=30)
# plt.title('Loss Contours around Trained Model Modified')
# # save plot to file
# plt.savefig('loss_mnist_2d_contour_mod.png')

model_initial = torchvision.models.resnet18(weights = None)
model_initial.fc = torch.nn.Linear(512, num_classes)
model_initial.eval()

model_final = operation.get_torchvision_model_list(subdataset_list, model_name, 10)[0]
model_final.eval()

x, y = operation.get_cifar10_for_one_model("original")
metric = sl_metrics.Loss(criterion, x, y)
loss_data = loss_landscapes.linear_interpolation(model_initial, model_final, metric, LOSS_STEPS, deepcopy_model=True)

plt.plot([1/LOSS_STEPS * i for i in range(LOSS_STEPS)], loss_data)
plt.title('Linear Interpolation of Loss')
plt.xlabel('Interpolation Coefficient')
plt.ylabel('Loss')
axes = plt.gca()
plt.savefig('loss_mnist_1d_contour_org.png')

# model = torchvision.models.resnet18(weights = None)
# # model.fc = torch.nn.Linear(512, num_classes)
# model_path = 'resnet18.pt'
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()

# x, y = operation.get_cifar10_for_one_model("original")
# metric = sl_metrics.LossTorchvision(criterion, x, y)
# loss_data_fin_this, model_info_this, dir_one, dir_two = loss_landscapes_main.random_plane(model, metric, 10, LOSS_STEPS, normalization='filter', deepcopy_model=True)

# # plot the training loss landscape
# plt.clf()
# plt.contour(loss_data_fin_this, levels=30)
# plt.title('Loss Contours around Trained Model PreTrained')
# # save plot to file
# plt.savefig('loss_mnist_2d_contour_pre.png')