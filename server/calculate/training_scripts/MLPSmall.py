import torch
import torch.nn.functional as F
import numpy as np

IN_DIM = 28 * 28
OUT_DIM = 10
HIDDEN_UNITS = 512


class MLPSmall(torch.nn.Module):
    """Fully connected feed-forward neural network with one hidden layer."""

    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(IN_DIM, HIDDEN_UNITS)
        self.linear_2 = torch.nn.Linear(HIDDEN_UNITS, OUT_DIM)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)


class Flatten(object):
    """Transforms a PIL image to a flat numpy array."""

    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()
