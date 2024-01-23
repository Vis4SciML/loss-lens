import sys
import argparse

sys.path.append("../")

# libraries
import copy
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
from torchmetrics import Accuracy
import random
import os
import json

model_id = "mnist_mlp"

parser = argparse.ArgumentParser(description="Train a MLP model on MNIST.")
parser.add_argument(
    "--seed", type=int, default=0, metavar="S", help="random seed", required=True
)
args = parser.parse_args()

# training hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 0.001
BATCH_SIZE = 1024
EPOCHS = 150
HIDDEN_UNITS = 512


# setting the seed
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class MLPSmall(torch.nn.Module):
    """Fully connected feed-forward neural network with one hidden layer."""

    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, HIDDEN_UNITS)
        self.linear_2 = torch.nn.Linear(HIDDEN_UNITS, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)


class Flatten(object):
    """Transforms a PIL image to a flat numpy array."""

    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()


def train(model, optimizer, criterion, train_loader, epochs):
    """Trains the given model with the given optimizer, loss function, etc."""
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[70, 110], gamma=0.1
    )
    model.train()
    # train model
    with tqdm(total=epochs, desc="Progress", unit="iter", ncols=100) as pbar:
        for epoch in range(epochs):
            for count, batch in enumerate(train_loader, 0):
                optimizer.zero_grad()
                x, y = batch
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                accuracy = Accuracy(task="multiclass", num_classes=10)(pred, y)
                loss = loss.item()
                if count % BATCH_SIZE == 0:
                    pbar.set_postfix(Accuracy=f"{accuracy:.2%}", Loss=f"{loss:.2f}")
            scheduler.step()
            pbar.update(1)
    model.eval()


# download MNIST and setup data loaders
mnist_train = datasets.MNIST(
    root="../data", train=True, download=True, transform=Flatten()
)
train_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=BATCH_SIZE, shuffle=True
)

# define model
model = MLPSmall(IN_DIM, OUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# save the model
path_dir = "../trained_models/" + model_id + "/"
if not os.path.exists(path_dir):
    os.makedirs(path_dir)

# stores the initial point in parameter space
# model_initial = copy.deepcopy(model)
# path = path_dir + model_id + '_initial.pt'
# torch.save(model_initial.state_dict(), path)

# train the model
train(model, optimizer, criterion, train_loader, EPOCHS)

model_final = copy.deepcopy(model)

path = path_dir + model_id + "_" + str(SEED) + ".pt"
torch.save(model_final.state_dict(), path)


model_info = {
    "modelId": model_id,
    "modelName": model_id,
    "modelDescription": "Learning Rate: "
    + str(LR)
    + ", Batch Size: "
    + str(BATCH_SIZE)
    + ", Epochs: "
    + str(EPOCHS)
    + ", Hidden Units: "
    + str(HIDDEN_UNITS),
    "modelDataset": "MNIST",
    "datasetId": "mnist",
    "modelDatasetDescription": "MNIST dataset 60000",
}


path = path_dir + "model_info.json"

if not os.path.exists(path):
    with open(path, "w") as f:
        json.dump(model_info, f)
