# libraries
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from numpy import load
import torchvision.datasets as datasets

# from pyevtk.hl import gridToVTK
from pyevtk.hl import imageToVTK

from operation import functions as operation

# mdoel hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 512
STEPS = 25
START = -0.5
END = 0.5

# Prediction distribution parameters
MAX_NUM = 100
INPUT_DATA_PERCENTAGE = 0.01

# training hyperparameters
dataset_name = 'MNIST'

class MLPSmall(torch.nn.Module):
    """Fully connected feed-forward neural network with one hidden layer."""
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.linear_2 = torch.nn.Linear(32, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)

class Flatten(object):
    """Transforms a PIL image to a flat numpy array."""
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()

parser = argparse.ArgumentParser()
# dataset for training and testing the model:
parser.add_argument('--model', default='CNN', help='trained model for testing')
parser.add_argument('--reshapex', default=397, help='reshape x for cka distance calculation') 
parser.add_argument('--reshapey', default=64, help='reshape y for cka distance calculation')
parser.add_argument('--losssteps', default=50, help='steps for loss landscape calculation')
args = parser.parse_args()

# Loss Landscape parameters
LOSS_STEPS = int(args.losssteps)
# Model Name
model_name = args.model
# set the subdataset list
subdataset_list = ['original', 'brightness', 'canny_edges', 'dotted_line', 'fog', 'glass_blur', 'impulse_noise', 'motion_blur', 'rotate', 'scale', 'shear', 'shot_noise', 'spatter', 'stripe', 'translate', 'zigzag', 'all']

# set the client and database name
client = "mongodb://localhost:27017/"
database_name = "mydatabase"
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

# define the loss function
criterion = torch.nn.CrossEntropyLoss()

# calculate the torch CKA similarity between layers
print("Calculating the CKA similarity between model layers...")
cka_result_all = operation.calculate_model_layer_torch_cka_similarity(dataset_name, subdataset_list, IN_DIM, OUT_DIM)
print("Finished calculating the CKA similarity between model layers.")

# print the torch CKA similarity between models
for i in range(len(cka_result_all)):
    cka_result = cka_result_all[i]
    for j in range(len(cka_result)):
        print("Torch CKA similarity between model " + str(i) + " and model " + str(j) + " is: ", cka_result[j]["CKA"])