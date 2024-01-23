import random
import time
from typing import Dict, List, Optional, Union
import sys
import os
import torch
import copy
import pickle
import torchvision.datasets as datasets
import torchvision
import torchmetrics
from pyhessian import hessian
import numpy as np
from sklearn.manifold import MDS
import collections
from vit_pytorch import ViT
import torch.nn as nn
from torch.types import Device
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from robustbench.data import load_cifar10c



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_dir + "/training_scripts/")
sys.path.append(parent_dir + "/training_scripts/pinn/pbc_examples/")

from training_scripts.MLPSmall import MLPSmall
from training_scripts.RESNET20 import resnet
from script_util.torch_cka import cka as torch_cka
from script_util.torch_cka import cka_pinn as torch_cka_pinn
from training_scripts.MLPSmall import Flatten
from pinn.pbc_examples.choose_optimizer import *
from pinn.pbc_examples.net_pbc import * 
from pinn.pbc_examples.utils import *
from pinn.pbc_examples.systems_pbc import *
from pinn.pyhessian import hessian_pinn

# import loss_landscapes
# import loss_landscapes.metrics
# from loss_landscapes.model_interface.model_parameters import ModelParameters
# from loss_landscapes.contrib.functions import SimpleWarmupCaller, SimpleLossEvalCaller, log_refined_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.set_device(0)


# https://arxiv.org/abs/1905.00414
class CKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(
            self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma))
        )

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)


def load_mode(model_id: str, mode_id: str):
    mode = None
    mode_path = (
        parent_dir
        + "/trained_models/"
        + model_id
        + "/"
        + model_id
        + "_"
        + mode_id
        + ".pt"
    )
    if model_id == "mnist_mlp" or model_id == "mnist_mlp_less_epoch":
        mode = MLPSmall()
        mode.load_state_dict(torch.load(mode_path, map_location=DEVICE))
        mode.eval()
        mode.to(DEVICE)
    elif model_id == "cifar10_vit" or model_id == "cifar10_augvit":
        mode = ViT(image_size = 32, patch_size = 4, num_classes = 10, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)
        mode.load_state_dict(torch.load(mode_path, map_location=DEVICE))
        mode.eval()
        mode.to(DEVICE)
    elif model_id == "cifar10_resnet20":
        mode_path = (
            parent_dir
            + "/trained_models/"
            + model_id
            + "/"
            + model_id
            + "_"
            + mode_id
            + ".pkl"
        )
        mode = resnet(num_classes=10,
               depth=20,
               residual_not=True,
               batch_norm_not=True)
        mode = torch.nn.DataParallel(mode)
        mode.load_state_dict(torch.load(mode_path, map_location=DEVICE))
        mode.eval()
        mode.to(DEVICE)
    elif model_id == "cifar10_resnet20_no_skip":
        mode_path = (
            parent_dir
            + "/trained_models/"
            + model_id
            + "/"
            + model_id
            + "_"
            + mode_id
            + ".pkl"
        )
        mode = resnet(num_classes=10,
               depth=20,
               residual_not=False,
               batch_norm_not=True)
        mode = torch.nn.DataParallel(mode)
        mode.load_state_dict(torch.load(mode_path, map_location=DEVICE))
        mode.eval()
        mode.to(DEVICE)
    elif model_id == "pinn_convection_beta1" or model_id == "pinn_convection_beta50":
        mode = torch.load(mode_path, map_location=DEVICE)
        mode.dnn.eval()
    else:
        raise ValueError("Model id not found.")

    return mode


def load_data(model_id: str, train: bool = False):
    if model_id == "mnist_mlp" or model_id == "mnist_mlp_less_epoch":
        data = datasets.MNIST(
            root=parent_dir + "/data", train=train, download=True, transform=Flatten()
        )
        test_loader = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=True)
        return test_loader
    elif model_id == "cifar10_vit":
        data = datasets.CIFAR10(
            root=parent_dir + "/data", train=train, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
        return test_loader
    elif model_id == "cifar10_augvit":
        train_set = torchvision.datasets.CIFAR10('../data/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                            ]))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        # print(len(train_loader)* 64)
        test_set = torchvision.datasets.CIFAR10('../data/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                            ]))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        # print(len(test_loader)* 64)
        train_total = len(train_loader)* 64
        test_total = len(test_loader)* 64
        batch_size_train = 64
        offset = 25600
        num_cifar10c = train_total + test_total + offset
        x,targets = load_cifar10c(n_examples=num_cifar10c, data_dir=parent_dir + '/data/CIFAR10-C')
        # print(x.size())
        y1 = [x[batch_size_train*i:batch_size_train*i + batch_size_train,:,:,:] for i in range(int((train_total + offset)/batch_size_train), int(num_cifar10c/batch_size_train))]
        y2 = [targets[batch_size_train*i:batch_size_train*i + batch_size_train] for i in range(int((train_total + offset)/batch_size_train), int(num_cifar10c/batch_size_train))]
        return zip(y1,y2)


    elif model_id == "cifar10_resnet20" or model_id == "cifar10_resnet20_no_skip":
        data = datasets.CIFAR10(
            root=parent_dir + "/data", train=train, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
            )]))
        test_loader = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False)
        return test_loader
    elif model_id == "pinn_convection_beta1" or model_id == "pinn_convection_beta50":
        # training data?
        xgrid = 256    
        x = np.linspace(0, 2*np.pi, xgrid, endpoint=False).reshape(-1, 1) # not inclusive
        t = np.linspace(0, 1, 100).reshape(-1, 1)
        X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
        X_star = np.hstack((X.flatten()[:, None].astype(np.float), T.flatten()[:, None].astype(np.float))) # all the x,t "test" data

        # remove initial and boundaty data from X_star
        t_noinitial = t[1:]
        # remove boundary at x=0
        x_noboundary = x[1:]
        X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
        X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

        # sample collocation points only from the interior (where the PDE is enforced)
        X_f_train = sample_random(X_star_noinitial_noboundary, 100)

        beta = 1
        # training labels?
        u_vals = convection_diffusion("sin(x)", 1.0, beta, 0, xgrid, 100)
        G = np.full(X_f_train.shape[0], float(0))

        u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)

        return (X_star, u_star)

    else:
        raise ValueError("Model id not found.")



def compute_mode_performance(model_id: str, mode_id: str) -> Dict[str, float]:
    mode = load_mode(model_id, mode_id)
    data = load_data(model_id, train=False)
    if model_id == "pinn_convection_beta1" or model_id == "pinn_convection_beta50":
        X_star, u_star = data
        u_pred = mode.predict(X_star)
        print(u_pred.shape)
        print(u_pred)
        print(u_star.shape)
        print(u_star)
        error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
        error_u_abs = np.mean(np.abs(u_star - u_pred))
        error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)

        print('Error u rel: %e' % (error_u_relative))
        print('Error u abs: %e' % (error_u_abs))
        print('Error u linf: %e' % (error_u_linf))

        return {
            "error_rel": error_u_relative,
            "error_abs": error_u_abs,
            "error_linf": error_u_linf,
        }

    else:
        accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(DEVICE)
        recall_metric = torchmetrics.Recall(task="multiclass", num_classes=10).to(DEVICE)
        precision_metric = torchmetrics.Precision(task="multiclass", num_classes=10).to(DEVICE)
        f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=10).to(DEVICE)

        with torch.no_grad():
            for inputs, labels in data:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = mode(inputs)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                accuracy_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
                precision_metric.update(predicted, labels)
                f1_metric.update(predicted, labels)

        accuracy = accuracy_metric.compute()
        accuracy = accuracy.item()

        precision = precision_metric.compute()
        precision = precision.item()

        recall = recall_metric.compute()
        recall = recall.item()

        f1 = f1_metric.compute()
        f1 = f1.item()

        performance = {}

        # compute accuracy
        performance["accuracy"] = accuracy

        # compute precision
        performance["precision"] = precision

        # compute recall
        performance["recall"] = recall

        # compute f1
        performance["f1"] = f1

        return performance


def compute_mode_hessian(model_id: str, mode_id: str) -> List[List[float]]:
    mode = load_mode(model_id, mode_id)

    criterion = torch.nn.CrossEntropyLoss()
    if model_id == "pinn_convection_beta1" or model_id == "pinn_convection_beta50":
        x = np.linspace(0, 2*np.pi, 256, endpoint=False).reshape(-1, 1) # not inclusive
        t = np.linspace(0, 1, 100).reshape(-1, 1)
        X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

        # remove initial and boundaty data from X_star
        t_noinitial = t[1:]
        # remove boundary at x=0
        x_noboundary = x[1:]
        X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
        X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

        # sample collocation points only from the interior (where the PDE is enforced)
        X_f_train = sample_random(X_star_noinitial_noboundary, 100)
        x, y = iter(X_f_train).__next__()
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(DEVICE)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(DEVICE)
        hessian_comp = hessian_pinn(mode, copy.deepcopy(mode.dnn), criterion, data=(x, t), cuda=torch.cuda.is_available())
    else:
        data = load_data(model_id, train=True)
        train_loader_iter = iter(data)
        x, y = train_loader_iter.__next__()
        hessian_comp = hessian(mode, criterion, data=(x, y), cuda=torch.cuda.is_available())

    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=10)
    top_eigenvalues.sort(reverse=True)

    return top_eigenvalues


def update_mode_losslandscape(case_id: str, model_id: str, mode_id: str):
    losslandscape, max_value, min_value = compute_mode_losslandscape(model_id, mode_id)

    if not dbExists():
        createDB()

    if not collectionExists(LOSS_LANDSCAPE):
        createCollection(LOSS_LANDSCAPE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}

    record = {"grid": losslandscape}
    addOrUpdateDocument(LOSS_LANDSCAPE, query, record)

    boundary_query = {"caseId": case_id}
    boundary_record = getDocument(MODEL_META_DATA, boundary_query)
    if boundary_record is None:
        boundary_record = {
            "caseId": case_id,
        }
    if "lossBounds" not in boundary_record:
        boundary_record["lossBounds"] = {
            "upperBound": max_value,
            "lowerBound": min_value,
        }
    else:
        boundary_record["lossBounds"]["upperBound"] = (
            max_value
            if max_value > boundary_record["lossBounds"]["upperBound"]
            else boundary_record["lossBounds"]["upperBound"]
        )
        boundary_record["lossBounds"]["lowerBound"] = (
            min_value
            if min_value < boundary_record["lossBounds"]["lowerBound"]
            else boundary_record["lossBounds"]["lowerBound"]
        )

    addOrUpdateDocument(MODEL_META_DATA, boundary_query, boundary_record)

def update_mode_losslandscape(case_id: str, model_id: str, mode_id: str):
    losslandscape, max_value, min_value = compute_mode_losslandscape(model_id, mode_id)

    if not dbExists():
        createDB()

    if not collectionExists(LOSS_LANDSCAPE):
        createCollection(LOSS_LANDSCAPE)

    query = {"caseId": case_id, "modelId": model_id, "modeId": mode_id}

    record = {"grid": losslandscape}
    addOrUpdateDocument(LOSS_LANDSCAPE, query, record)

    boundary_query = {"caseId": case_id}
    boundary_record = getDocument(MODEL_META_DATA, boundary_query)
    if boundary_record is None:
        boundary_record = {
            "caseId": case_id,
        }
    if "lossBounds" not in boundary_record:
        boundary_record["lossBounds"] = {
            "upperBound": max_value,
            "lowerBound": min_value,
        }
    else:
        boundary_record["lossBounds"]["upperBound"] = (
            max_value
            if max_value > boundary_record["lossBounds"]["upperBound"]
            else boundary_record["lossBounds"]["upperBound"]
        )
        boundary_record["lossBounds"]["lowerBound"] = (
            min_value
            if min_value < boundary_record["lossBounds"]["lowerBound"]
            else boundary_record["lossBounds"]["lowerBound"]
        )

    addOrUpdateDocument(MODEL_META_DATA, boundary_query, boundary_record)


# def compute_mode_losslandscape(model_id: str, mode_id: str, criterion: Union[_Loss, nn.Module], 
#                                device: Union[None, Device] = None, 
#                                random_plain=True, random='normal', 
#                                dir_one:Union[None, ModelParameters]=None, 
#                                dir_two:Union[None, ModelParameters]=None, 
#                                steps=20, distance=1, normalization='layer', 
#                                warmup=False, log_refined=False, 
#                                centered=True, deepcopy_model=True):
#     
#     """
#     Compute the 2D loss landscape given the model, data and loss function.
#
#     :param model_id: The model id
#     :param mode_id: The mode id
#     :param criterion: The loss function used to evaluate losses
#     :param device: CPU or GPU used for computation
#     :param random_plain: To create loss landscape along random direction or given directions
#     :param random: The mode to generate random directions, must be one of 'normal', 'uniform'; Only available when random_plain is set to True
#     :param dir_one: The first given direction in parameter space; Only available when random_plain is set to False
#     :param dir_one: The second given direction in parameter space; Only available when random_plain is set to False
#     :param steps: How many steps it takes to evaluate from start to end
#     :param distance: The maximum distance in parameter space from the start point
#     :param normalization: The way to normalize parameters, must be one of 'filter', 'layer', 'model'
#     :param warmup: The indicator of whether using warmup to update running statistics before compute the loss lanscapes; Set it as True, if there are batchnorm (and other normalization) layers in the model and the distance is comparably large, otherwise, Set it as False
#     :param log_refined: The indicator of whether using log function to refine the loss lanscapes; Set it as True, if there are extreme values in the loss lanscape, otherwise, Set it as False
#     :param centered: If True, the loaded model is at the center of the loss lanscapes, otherwise, at the corner
#     :param deepcopy_model: If models are deepcopied in computation
#     
#     :return losslandscape: 2D numpy array of size (steps, steps) of the loss landscape
#     :return max_value: max_value of the loss landscape
#     :return min_value: min_value of the loss landscape
#     """
#     
#     # fetch model and data
#     mode = load_mode(model_id, mode_id)
#     data = load_data(model_id)
#     device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     
#     # instantiate a 2d losslandscape calculator object
#     pll = loss_landscapes.PlanarLossLandscape(mode, steps, deepcopy_model=deepcopy_model)
#     
#     if random_plain:
#         # set up random plain loss landscape computation
#         pll.random_plain(distance=distance, normalization=normalization, random=random, centered=centered)
#     else:
#         try:
#             if dir_one is None:
#                 raise TypeError('Parameter direction one cannot be None.')
#             if dir_two is None:
#                 raise TypeError('Parameter direction two cannot be None.')
#         except TypeError as e:
#             print('Directions in ModelParameters type should be provided.')
#         # set up loss landscape computation with given precomputed directions
#         pll.precomputed(dir_one, dir_two, distance=distance, normalization=normalization, centered=centered)
#     
#     pll.stats_initializer()
#     
#     if isinstance(data, DataLoader):
#         # dataloader loss landscape
#         losslandscape = np.zeros((steps, steps))
#         for i in range(steps):
#             for j in range(steps):
#                 if warmup:
#                     warm_up_caller = SimpleWarmupCaller(data, device)
#                     pll.outer_warm_up(i, j, warm_up_caller)
#                 loss_eval_caller = SimpleLossEvalCaller(data, criterion, device)
#                 losslandscape[i] = pll.outer_compute(i, j, loss_eval_caller)
#     elif isinstance(data, List):
#         # single batch loss landscape
#         x, y, *_ = iter(data).__next__()
#         metric = loss_landscapes.metrics.Loss(criterion, x.to(device), y.to(device))
#         pll.warm_up(metric)
#         loss_data_fin = pll.compute(metric)
#     else:
#         raise TypeError('The data is neither a dataloader nor a batch from dataloader.')
#         
#     # losslandscape = [[random.random() for i in range(40)] for j in range(40)]
#     if log_refined:
#         losslandscape = log_refined_loss(losslandscape)
#
#     # get the max value
#     # max_value = max([max(row) for row in losslandscape])
#     max_value = np.max(losslandscape)
#     # get the min value
#     # min_value = min([min(row) for row in losslandscape])
#     min_value = np.min(losslandscape)
#
#     return losslandscape, max_value, min_value
#

def compute_mode_merge_tree(model_id: str, mode_id: str) -> List[Dict[str, float]]:
    """
    TODO
    """
    # mode = load_mode(model_id, mode_id)
    # data = load_data(model_id)
    nodes = [{"x": random.random(), "y": random.random(), "id": i} for i in range(100)]
    edges = [
        {
            "sourceX": random.random(),
            "sourceY": random.random(),
            "targetX": random.random(),
            "targetY": random.random(),
        }
        for i in range(100)
    ]
    merge_tree = {"nodes": nodes, "edges": edges}

    return merge_tree


def compute_mode_persistence_barcode(
    model_id: str, mode_id: str
) -> List[Dict[str, float]]:
    """
    TODO
    """
    # mode = load_mode(model_id, mode_id)
    # data = load_data(model_id)
    persistence_barcode = [
        {"x": random.random(), "y0": random.random(), "y1": random.random()}
        for i in range(100)
    ]

    return persistence_barcode


def compute_cka_similarity(
    model0_id: str, model1_id: str, mode0_id: str, mode1_id: str
) -> float:
    mode0 = load_mode(model0_id, mode0_id)
    mode1 = load_mode(model1_id, mode1_id)
    if model0_id == 'mnist_mlp' or model0_id == 'mnist_mlp_less_epoch':
        flatten_mode0 = torch.cat(
            (mode0.linear_1.weight.data.reshape(-1), mode0.linear_2.weight.data.reshape(-1))
        )
        flatten_mode1 = torch.cat(
            (mode1.linear_1.weight.data.reshape(-1), mode1.linear_2.weight.data.reshape(-1))
        )
        flatten_mode0 = flatten_mode0.numpy().reshape((512, 794))
        flatten_mode1 = flatten_mode1.numpy().reshape((512, 794))

    elif model0_id == "cifar10_vit" or model0_id == "cifar10_augvit":
        flatten_mode0_params = []
        flatten_mode1_params = []
        for name, param in mode0.named_parameters():
            if "mlp_head" in name or "cls_token" in name or "norm" in name or "pos" in name:
                continue
            flatten_mode0_params.append(param.data.reshape(-1))

        for name, param in mode1.named_parameters():
            if "mlp_head" in name or "cls_token" in name or "norm" in name or "pos" in name:
                continue
            flatten_mode1_params.append(param.data.reshape(-1))

        flatten_mode0 = torch.cat(flatten_mode0_params)
        flatten_mode1 = torch.cat(flatten_mode1_params)

        # flatten_mode0 = torch.cat((mode0.to_patch_embedding[1].weight.reshape(-1),
        #                            mode0.to_patch_embedding[2].weight.reshape(-1),
        #                            mode0.to_patch_embedding[3].weight.reshape(-1),
        #                            mode0.mlp_head[0].weight.reshape(-1),
        #                            mode0.mlp_head[1].weight.reshape(-1)))
        # flatten_mode1 = torch.cat((mode1.to_patch_embedding[1].weight.reshape(-1),
        #                            mode1.to_patch_embedding[2].weight.reshape(-1),
        #                            mode1.to_patch_embedding[3].weight.reshape(-1),
        #                            mode1.mlp_head[0].weight.reshape(-1), 
        #                            mode1.mlp_head[1].weight.reshape(-1)))

        flatten_mode0 = flatten_mode0.detach().numpy().reshape((7193, 7008))
        flatten_mode1 = flatten_mode1.detach().numpy().reshape((7193, 7008))

    elif model0_id == "cifar10_resnet20" or model0_id == "cifar10_resnet20_no_skip":
        flatten_mode0 = torch.cat((mode0.module.conv1.weight.reshape(-1),
                                  mode0.module.bn1.weight.reshape(-1),
                                  mode0.module.fc.weight.reshape(-1)))
        flatten_mode1 = torch.cat((mode1.module.conv1.weight.reshape(-1),
                                  mode1.module.bn1.weight.reshape(-1),
                                  mode1.module.fc.weight.reshape(-1)))

        flatten_mode0 = flatten_mode0.detach().numpy().reshape((34, 32))
        flatten_mode1 = flatten_mode1.detach().numpy().reshape((34, 32))
    elif model0_id == "pinn_convection_beta1" or model0_id == "pinn_convection_beta50":
        mode0 = mode0.dnn
        mode1 = mode1.dnn
        flatten_mode0 = torch.cat((mode0.layers.layer_0.weight.reshape(-1),
                                  mode0.layers.layer_1.weight.reshape(-1),
                                  mode0.layers.layer_2.weight.reshape(-1),
                                  mode0.layers.layer_3.weight.reshape(-1),
                                  mode0.layers.layer_4.weight.reshape(-1),
                                   ))
        flatten_mode1 = torch.cat((mode1.layers.layer_0.weight.reshape(-1),
                                  mode1.layers.layer_1.weight.reshape(-1),
                                  mode1.layers.layer_2.weight.reshape(-1),
                                  mode1.layers.layer_3.weight.reshape(-1),
                                  mode1.layers.layer_4.weight.reshape(-1),
                                   ))
        flatten_mode0 = flatten_mode0.detach().numpy().reshape((51, 150))
        flatten_mode1 = flatten_mode1.detach().numpy().reshape((51, 150))
        


    np_cka = CKA()
    cka_res = np_cka.linear_CKA(flatten_mode0, flatten_mode1)
    # print(cka_res)

    return cka_res



def compute_layer_similarity(
    model0_id: str, model1_id: str, mode0_id: str, mode1_id: str
) -> List[List[float]]:
    mode0 = load_mode(model0_id, mode0_id)
    mode1 = load_mode(model1_id, mode1_id)


    data = load_data(model0_id, train=True)

    if model0_id == 'pinn_convection_beta1' or model0_id == 'pinn_convection_beta50':
        mode0 = mode0
        mode1 = mode1
        data, _ = data
        # data = data.astype(np.float)
        data = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False)
        cka = torch_cka_pinn.CKA_PINN(mode0, mode1, device=DEVICE)
        cka.compare(data)
        results = cka.export()
    else:
        cka = torch_cka.CKA(mode0, mode1, device=DEVICE)
        cka.compare(data)
        results = cka.export()




    layer_similarity = results["CKA"].tolist()

    # layer_similarity = [[random.random() for i in range(100)] for j in range(100)]

    return layer_similarity


def compute_mode_connectivity(model_id: str, mode0_id: str, mode1_id: str) -> float:
    """
    TODO
    """
    # mode = load_mode(model_id, mode_id)
    # data = load_data(model_id)
    connectivity = random.random()

    return connectivity


def compute_confusion_matrix(
    model0_id: str, model1_id: str, mode0_id: str, mode1_id: str
) -> List[List[float]]:
    mode0 = load_mode(model0_id, mode0_id)
    mode1 = load_mode(model1_id, mode1_id)
    data = load_data(model0_id, train=False)
    confusion_matrix0 = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=10)
    confusion_matrix1 = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=10)
    with torch.no_grad():
        for x, y in data:
            y_pred0 = mode0(x)
            y_pred1 = mode1(x)

            _, y_pred0 = torch.max(y_pred0, 1)
            _, y_pred1 = torch.max(y_pred1, 1)
            confusion_matrix0.update(y_pred0, y)
            confusion_matrix1.update(y_pred1, y)

    confusion_matrix0 = confusion_matrix0.compute()
    confusion_matrix1 = confusion_matrix1.compute()
    confusion_matrix0 = confusion_matrix0.cpu().numpy()
    confusion_matrix1 = confusion_matrix1.cpu().numpy()

    res = []

    for i in range(len(confusion_matrix0)):
        cm = {
            "tp": [int(confusion_matrix0[i][i]), int(confusion_matrix1[i][i])],
            "fp": [
                int(sum(confusion_matrix0[:, i]) - confusion_matrix0[i][i]),
                int(sum(confusion_matrix1[:, i]) - confusion_matrix1[i][i]),
            ],
            "fn": [
                int(sum(confusion_matrix0[i]) - confusion_matrix0[i][i]),
                int(sum(confusion_matrix1[i]) - confusion_matrix1[i][i]),
            ],
            "tn": [
                int(
                    sum(sum(confusion_matrix0))
                    - sum(confusion_matrix0[i])
                    - sum(confusion_matrix0[:, i])
                    + confusion_matrix0[i][i]
                ),
                int(
                    sum(sum(confusion_matrix1))
                    - sum(confusion_matrix1[i])
                    - sum(confusion_matrix1[:, i])
                    + confusion_matrix1[i][i]
                ),
            ],
        }

        res.append(cm)

    class_names = ["class" + str(i) for i in range(10)]
    return res, class_names


def compute_position(distances) -> List[float]:
    res = {}
    # generate the distance matrix for linear CKA
    linear_cka_similarity_column = [d["ckaSimilarity"] for d in distances]
    # calculate the MDS of the linear CKA distance matrix for models similarity
    linear_cka_mds = MDS(n_components=2, dissimilarity="precomputed")

    unique_mode_ids = set()

    for distance in distances:
        id = distance["model0Id"] + "-" + distance["mode0Id"]
        unique_mode_ids.add(id)
        id = distance["model1Id"] + "-" + distance["mode1Id"]
        unique_mode_ids.add(id)

    linear_cka_matrix = [
        [0] * len(unique_mode_ids) for i in range(len(unique_mode_ids))
    ]

    unique_mode_ids = list(unique_mode_ids)
    mode_index_map = {}
    for i, mode_id in enumerate(unique_mode_ids):
        mode_index_map[mode_id] = i

    for distance in distances:
        index0 = mode_index_map[distance["model0Id"] + "-" + distance["mode0Id"]]
        index1 = mode_index_map[distance["model1Id"] + "-" + distance["mode1Id"]]
        linear_cka_matrix[index0][index1] = distance["ckaSimilarity"]
        linear_cka_matrix[index1][index0] = distance["ckaSimilarity"]

    linear_cka_matrix = 1 - np.array(linear_cka_matrix)
    linear_cka_embedding = linear_cka_mds.fit_transform(linear_cka_matrix)
    res = collections.defaultdict(dict)
    for i, pos in enumerate(linear_cka_embedding):
        res[unique_mode_ids[i]]["x"] = pos[0]
        res[unique_mode_ids[i]]["y"] = pos[1]

    return res


def get_model_layer_names(modelId: str):
    # TODO
    pass

