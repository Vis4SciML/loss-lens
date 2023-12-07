import abc
import typing

import torch
import torch.nn as nn
from torch.types import Device
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
import numpy as np

from loss_landscapes.model_interface.model_wrapper import ModelWrapper

def log_refined_loss(loss):
    return np.log(1.+loss)

class SimpleWarmupCaller(object):
    def __init__(self, data_loader: DataLoader, device: typing.Union[None, Device] = None, start=0):
        self.data_loader = data_loader
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start = start

    def __call__(self, model: ModelWrapper):
        model.train()
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader, self.start):
                x, y = batch
                x = x.to(self.device)
                model.forward(x)

class SimpleLossEvalCaller(object):
    def __init__(self, data_loader: DataLoader, criterion: nn.Module, device: typing.Union[None, Device] = None, start=0):
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start = start

    def __call__(self, model: ModelWrapper):
        model.eval()
        count = 0
        loss = 0.
        with torch.no_grad():
            for count, batch in enumerate(self.data_loader, self.start):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.shape[0]

                pred = model.forward(x)
                batch_loss = self.criterion(pred, y)
                                
                loss = count/(count+batch_size)*loss + batch_size/(count+batch_size)*batch_loss.item()
                count+=batch_size
        return loss