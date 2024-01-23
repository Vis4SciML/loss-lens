"""
A library of pre-written evaluation functions for PyTorch loss functions.

The classes and functions in this module cover common loss landscape evaluations. In particular,
computing the loss, the gradient of the loss (w.r.t. model parameters) and Hessian of the loss
(w.r.t. model parameters) for some supervised learning loss is easily accomplished.
"""


import numpy as np
import torch
import torch.autograd
from loss_landscapes.metrics.metric import Metric
from loss_landscapes.model_interface.model_parameters import rand_u_like, rand_n_like
from loss_landscapes.model_interface.model_wrapper import ModelWrapper


class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()


class LossGradient(Metric):
    """ Computes the gradient of a specified loss function w.r.t. the model parameters
    over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> np.ndarray:
        loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target)
        gradient = torch.autograd.grad(loss, model_wrapper.named_parameters()).detach().numpy()
        model_wrapper.zero_grad()
        return gradient


class LossPerturbations(Metric):
    """ Computes random perturbations in the loss value along a sample or random directions.
    These perturbations can be used to reason probabilistically about the curvature of a
    point on the loss landscape, as demonstrated in the paper by Schuurmans et al
    (https://arxiv.org/abs/1811.11214)."""
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor, n_directions: int, alpha: float, random='uniform'):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target
        self.n_directions = n_directions
        self.alpha = alpha
        self.random = random

    def __call__(self, model_wrapper: ModelWrapper) -> np.ndarray:
        # start point and directions
        start_point = model_wrapper.get_module_parameters()
        start_loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()

        # compute start loss and perturbed losses
        results = []
        for idx in range(self.n_directions):
            if self.random == 'uniform':
                direction = rand_u_like(start_point)
            elif self.random == 'normal':
                direction = rand_n_like(start_point)
            else:
                raise AttributeError('Unsupported random argument. Supported values are uniform and normal')

            start_point.add_(direction)

            loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()
            results.append(loss - start_loss)

            start_point.sub_(direction)

        return np.array(results)
