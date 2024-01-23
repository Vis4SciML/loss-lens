# libraries
import numpy as np
import torch
from numpy import load
import torchvision.datasets as datasets

from operation import functions as operation
from matplotlib import pyplot as plt

# mdoel hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 512
STEPS = 25
START = -0.5
END = 0.5

# Loss Landscape parameters
LOSS_STEPS = 40

# dataset name
dataset_name = 'MNIST'

# Prediction distribution parameters
INPUT_DATA_PERCENTAGE = 0.01

class Flatten(object):
    """Transforms a PIL image to a flat numpy array."""
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()

# set the client and database name
client = "mongodb://localhost:27017/"
database_name = "mydatabase"
# set all the collection names
collection_name_labels = "prediction_distribution_labels"

# data used for prediction distribution
print("Loading MNIST and MNIST-C dataset for prediction distribution...")
# data used for prediction distribution
mnist_original = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
# define the data loader
mnist_original_test_loader_prediction_distribution = torch.utils.data.DataLoader(mnist_original, batch_size=1, shuffle=False)
# set iterator to data loader
iterator = iter(mnist_original_test_loader_prediction_distribution)
# define the result list
mnist_test_prediction_distribution = []
for i in range(int(len(mnist_original_test_loader_prediction_distribution)*INPUT_DATA_PERCENTAGE)):
    x, y = iterator.__next__()
    mnist_test_prediction_distribution.append([x, y])
x_c = load('../data/MNIST_C/brightness/train_images.npy')
y_c = load('../data/MNIST_C/brightness/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/canny_edges/train_images.npy')
y_c = load('../data/MNIST_C/canny_edges/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/dotted_line/train_images.npy')
y_c = load('../data/MNIST_C/dotted_line/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/fog/train_images.npy')
y_c = load('../data/MNIST_C/fog/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/glass_blur/train_images.npy')
y_c = load('../data/MNIST_C/glass_blur/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/impulse_noise/train_images.npy')
y_c = load('../data/MNIST_C/impulse_noise/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/motion_blur/train_images.npy')
y_c = load('../data/MNIST_C/motion_blur/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/rotate/train_images.npy')
y_c = load('../data/MNIST_C/rotate/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/scale/train_images.npy')
y_c = load('../data/MNIST_C/scale/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/shear/train_images.npy')
y_c = load('../data/MNIST_C/shear/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/shot_noise/train_images.npy')
y_c = load('../data/MNIST_C/shot_noise/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/spatter/train_images.npy')
y_c = load('../data/MNIST_C/spatter/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/stripe/train_images.npy')
y_c = load('../data/MNIST_C/stripe/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/translate/train_images.npy')
y_c = load('../data/MNIST_C/translate/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/zigzag/train_images.npy')
y_c = load('../data/MNIST_C/zigzag/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(int(len(x_c)*INPUT_DATA_PERCENTAGE)):
    mnist_test_prediction_distribution.append([x_c[i], y_c[i]])
print("Finish loading the testing data for prediction distribution calculation.")

# define testing data loader for prediction distribution calculation
distribution_test_loader = torch.utils.data.DataLoader(mnist_test_prediction_distribution, batch_size=1, shuffle=False)

# set iterator to test_loader
test_loader_iter = iter(distribution_test_loader)

# transfer and save images and labels
mnist_prediction_distribution_labels = []

# save the images in local directory
for i in range(len(distribution_test_loader)):
    x, y = test_loader_iter.__next__()
    mnist_prediction_distribution_labels.append(y.item())
    x = x.reshape((28, 28))
    plt.clf()
    plt.axis('off')
    plt.imshow(x, cmap='gray')
    plt.savefig("MNIST_figures/figure_" + str(i) + ".png", bbox_inches='tight')
    print("saved " + str(i+1) + " image")

# save the labels into database
record = {"dataset": dataset_name, "labels": mnist_prediction_distribution_labels}
record_id = operation.store_into_database(client, database_name, collection_name_labels, record)
print("One record of prediction_distribution_labels has been successfully inserted with ID " + str(record_id))