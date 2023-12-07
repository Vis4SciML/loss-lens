# libraries
import numpy as np
from tqdm import tqdm
import numpy as np
import math
import copy
import torch
from numpy import load
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from sklearn.manifold import MDS
from pyhessian import hessian
from tsne_torch import TorchTSNE as TSNE
from torchmetrics import Accuracy, Recall, Precision, F1Score, ConfusionMatrix
from pytorchcv.model_provider import get_model as ptcv_get_model
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import to_tree, ClusterNode, dendrogram
from scipy.spatial.distance import squareform
from string import ascii_lowercase
from typing import Dict, Tuple, List, Union, Optional
import numpy.ma as ma
from vit_pytorch import ViT
from robustbench.data import load_cifar10c
import json

from database import functions

import operation.loss_landscapes as loss_landscapes
import operation.loss_landscapes.metrics as metrics
import operation.loss_landscapes.metrics.sl_metrics as sl_metrics
import operation.loss_landscapes.main as loss_landscapes_main
import operation.torch_cka.cka as torch_cka

# Variables
CIFAR10_BATCH_SIZE = 64
CIFAR10_SIZE = 60000
CIFAR10C_PERCENTAGE_LAYER_CKA = 0.001
CIFAR10_BATCH_SIZE_LAYER_CKA = 10
CIFAR10C_PERCENTAGE = 1
MNIST_BATCH_SIZE = 512
MNIST_PERCENTAGE_LAYER_CKA = 0.005
MNIST_BATCH_SIZE_LAYER_CKA = 10

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLAG = True if torch.cuda.is_available() else False

class MLPSmall(torch.nn.Module):
    """Fully connected feed-forward neural network with one hidden layer."""
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.linear_2 = torch.nn.Linear(32, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)

class MLPSmallModify(torch.nn.Module):
    """Fully connected feed-forward neural network with one hidden layer."""
    def __init__(self, x_dim, y_dim, linear1_weight, linear2_weight, linear1_bias, linear2_bias):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.linear_2 = torch.nn.Linear(32, y_dim)
        self.linear_1.weight = linear1_weight
        self.linear_2.weight = linear2_weight
        self.linear_1.bias = linear1_bias
        self.linear_2.bias = linear2_bias
    
    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)

class Flatten(object):
    """Transforms a PIL image to a flat numpy array."""
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()

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
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

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

class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def Hbeta(D=np.array([]), beta=1.0):
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def pca(X=np.array([]), no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4. # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

def _scipy_tree_to_newick_list(node: ClusterNode, newick: List[str], parentdist: float, leaf_names: List[str]) -> List[str]:
    if node.is_leaf():
        return newick + [f'{leaf_names[node.id]}:{parentdist - node.dist}']

    if len(newick) > 0:
        newick.append(f'):{parentdist - node.dist}')
    else:
        newick.append(');')
    newick = _scipy_tree_to_newick_list(node.get_left(), newick, node.dist, leaf_names)
    newick.append(',')
    newick = _scipy_tree_to_newick_list(node.get_right(), newick, node.dist, leaf_names)
    newick.append('(')
    return newick

def to_newick(tree: ClusterNode, leaf_names: List[str]) -> str:
    newick_list = _scipy_tree_to_newick_list(tree, [], tree.dist, leaf_names)
    return ''.join(newick_list[::-1])

def calculate_model_similarity_global_structure(dataset, subdataset_list, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # join the tensors of all subdatasets
    modelTensor_list = []
    for model in model_list:
        modelTensor_list.append(torch.cat((model.linear_1.weight.data.reshape(-1),model.linear_2.weight.data.reshape(-1))))
    
    # calculate the euclidean distance between models
    euclidean_distance_column = []
    for i in range(len(model_list)):
        euclidean_distance_row = []
        for j in range(len(model_list)):
            euclidean_distance_row.append((modelTensor_list[i] - modelTensor_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column.append(euclidean_distance_row)

    # generate the distance matrix
    euclidean_distance_matrix = np.array(euclidean_distance_column)
    
    # calculate the MDS of the euclidean distance matrix for models similarity
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(euclidean_distance_matrix)

    # calculate the global structure of the models based on the MDS
    condensed_distance_matrix = pdist(X_transformed)
    Z = hierarchy.linkage(condensed_distance_matrix, 'single')
    T = to_tree(Z, rd=False)
    figure  = to_newick(T, ascii_lowercase)

    return X_transformed, figure

def calculate_resnet_euclidean_distance_similarity(subdataset_list, model_depth, residual, batch_norm):
    # prepare all the models
    model_list = get_resnet_model_list(subdataset_list, model_depth, residual, batch_norm)

    # join the tensors of all subdatasets
    modelTensor_list = []
    for model in model_list:
        modelTensor_list.append(torch.cat((model.module.conv1.weight.data.reshape(-1),model.module.bn1.weight.data.reshape(-1),model.module.fc.weight.data.reshape(-1))))
    
    # calculate the euclidean distance between models
    euclidean_distance_column = []
    for i in range(len(model_list)):
        euclidean_distance_row = []
        for j in range(len(model_list)):
            euclidean_distance_row.append((modelTensor_list[i] - modelTensor_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column.append(euclidean_distance_row)

    # generate the distance matrix
    euclidean_distance_matrix = np.array(euclidean_distance_column)
    
    # calculate the MDS of the euclidean distance matrix for models similarity
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(euclidean_distance_matrix)

    return X_transformed

def calculate_model_cka_similarity_global_structure(dataset, subdataset_list, IN_DIM, OUT_DIM, reshape_x, reshape_y):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # join the tensors of all subdatasets
    modelTensor_list = []
    for model in model_list:
        modelTensor_list.append(torch.cat((model.linear_1.weight.data.reshape(-1),model.linear_2.weight.data.reshape(-1))))
    
    # initialize the CKA class
    np_cka = CKA()
    
    # calculate the linear CKA similarity between models
    linear_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        linear_cka_similarity_row = []
        for j in range(len(model_list)):
            linear_cka_similarity_row.append(np_cka.linear_CKA(modelTensor_list[i].numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].numpy().reshape((reshape_x, reshape_y))))
        linear_cka_similarity_column.append(linear_cka_similarity_row)
    
    kernel_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        kernel_cka_similarity_row = []
        for j in range(len(model_list)):
            kernel_cka_similarity_row.append(np_cka.kernel_CKA(modelTensor_list[i].numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].numpy().reshape((reshape_x, reshape_y))))
        kernel_cka_similarity_column.append(kernel_cka_similarity_row)
    
    # generate the distance matrix for linear CKA
    linear_cka_matrix = 1 - np.array(linear_cka_similarity_column)
    
    # generate the distance matrix for RBF kernel CKA
    kernel_cka_matrix = 1 - np.array(kernel_cka_similarity_column)
    
    # calculate the MDS of the linear CKA distance matrix for models similarity
    linear_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    linear_cka_embedding = linear_cka_mds.fit_transform(linear_cka_matrix)

    # calculate the global structure of the models based on the linear CKA MDS
    linear_cka_condensed_distance_matrix = pdist(linear_cka_embedding)
    linear_cka_Z = hierarchy.linkage(linear_cka_condensed_distance_matrix, 'single')
    linear_cka_T = to_tree(linear_cka_Z, rd=False)
    linear_cka_figure  = to_newick(linear_cka_T, ascii_lowercase)

    # calculate the MDS of the RBF kernel CKA distance matrix for models similarity
    kernel_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    kernel_cka_embedding = kernel_cka_mds.fit_transform(kernel_cka_matrix)

    # calculate the global structure of the models based on the RBF kernel CKA MDS
    kernel_cka_condensed_distance_matrix = pdist(kernel_cka_embedding)
    kernel_cka_Z = hierarchy.linkage(kernel_cka_condensed_distance_matrix, 'single')
    kernel_cka_T = to_tree(kernel_cka_Z, rd=False)
    kernel_cka_figure  = to_newick(kernel_cka_T, ascii_lowercase)

    return linear_cka_embedding, kernel_cka_embedding, linear_cka_figure, kernel_cka_figure

def calculate_resnet_cka_similarity(subdataset_list, model_depth, residual, batch_norm, reshape_x, reshape_y):
    # prepare all the models
    model_list = get_resnet_model_list(subdataset_list, model_depth, residual, batch_norm)

    # join the tensors of all subdatasets
    modelTensor_list = []
    for model in model_list:
        modelTensor_list.append(torch.cat((model.module.conv1.weight.data.reshape(-1),model.module.bn1.weight.data.reshape(-1),model.module.fc.weight.data.reshape(-1))))

    # initialize the CKA class
    np_cka = CKA()
    
    # calculate the linear CKA similarity between models
    linear_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        linear_cka_similarity_row = []
        for j in range(len(model_list)):
            linear_cka_similarity_row.append(np_cka.linear_CKA(modelTensor_list[i].numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].numpy().reshape((reshape_x, reshape_y))))
        linear_cka_similarity_column.append(linear_cka_similarity_row)
    
    kernel_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        kernel_cka_similarity_row = []
        for j in range(len(model_list)):
            kernel_cka_similarity_row.append(np_cka.kernel_CKA(modelTensor_list[i].numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].numpy().reshape((reshape_x, reshape_y))))
        kernel_cka_similarity_column.append(kernel_cka_similarity_row)
    
    # generate the distance matrix for linear CKA
    linear_cka_matrix = 1 - np.array(linear_cka_similarity_column)
    
    # generate the distance matrix for RBF kernel CKA
    kernel_cka_matrix = 1 - np.array(kernel_cka_similarity_column)
    
    # calculate the MDS of the linear CKA distance matrix for models similarity
    linear_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    linear_cka_embedding = linear_cka_mds.fit_transform(linear_cka_matrix)
    
    # calculate the MDS of the RBF kernel CKA distance matrix for models similarity
    kernel_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    kernel_cka_embedding = kernel_cka_mds.fit_transform(kernel_cka_matrix)

    return linear_cka_embedding, kernel_cka_embedding

def calculate_layer_euclidean_distance_similarity(dataset, subdataset_list, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # get the layer weights of all models
    layerOne_list = []
    layerTwo_list = []
    for model in model_list:
        layerOne_list.append(model.linear_1.weight.data)
        layerTwo_list.append(model.linear_2.weight.data)
    
    # calculate the euclidean distance for the first layer
    euclidean_distance_column_layerOne = []
    for i in range(len(layerOne_list)):
        euclidean_distance_row_layerOne = []
        for j in range(len(layerOne_list)):
            euclidean_distance_row_layerOne.append((layerOne_list[i] - layerOne_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column_layerOne.append(euclidean_distance_row_layerOne)
    
    # generate the distance matrix
    euclidean_distance_matrix_layerOne = np.array(euclidean_distance_column_layerOne)
    
    # calculate the euclidean distance for the second layer
    euclidean_distance_column_layerTwo = []
    for i in range(len(layerTwo_list)):
        euclidean_distance_row_layerTwo = []
        for j in range(len(layerTwo_list)):
            euclidean_distance_row_layerTwo.append((layerTwo_list[i] - layerTwo_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column_layerTwo.append(euclidean_distance_row_layerTwo)
    
    # generate the distance matrix
    euclidean_distance_matrix_layerTwo = np.array(euclidean_distance_column_layerTwo)
    
    # calculate the MDS of the euclidean distance matrix for layers similarity
    embedding_layerOne = MDS(n_components=1,dissimilarity='precomputed')
    X_transformed_layerOne = embedding_layerOne.fit_transform(euclidean_distance_matrix_layerOne)
    embedding_layerTwo = MDS(n_components=1,dissimilarity='precomputed')
    X_transformed_layerTwo = embedding_layerTwo.fit_transform(euclidean_distance_matrix_layerTwo)

    return X_transformed_layerOne, X_transformed_layerTwo, euclidean_distance_matrix_layerOne, euclidean_distance_matrix_layerTwo
    
def calculate_model_layer_torch_cka_similarity(dataset, subdataset_list, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # prepare the dataloader
    mnist_original = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
    mnist_original_train_loader = torch.utils.data.DataLoader(mnist_original, batch_size=1, shuffle=False)
    x, y = iter(mnist_original_train_loader).__next__()
    mnist_test = []
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x[i], y[i]])
    x_c = load('../data/MNIST_C/brightness/train_images.npy')
    y_c = load('../data/MNIST_C/brightness/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x_c)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/canny_edges/train_images.npy')
    y_c = load('../data/MNIST_C/canny_edges/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x_c)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/dotted_line/train_images.npy')
    y_c = load('../data/MNIST_C/dotted_line/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/fog/train_images.npy')
    y_c = load('../data/MNIST_C/fog/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/glass_blur/train_images.npy')
    y_c = load('../data/MNIST_C/glass_blur/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/impulse_noise/train_images.npy')
    y_c = load('../data/MNIST_C/impulse_noise/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/motion_blur/train_images.npy')
    y_c = load('../data/MNIST_C/motion_blur/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/rotate/train_images.npy')
    y_c = load('../data/MNIST_C/rotate/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/scale/train_images.npy')
    y_c = load('../data/MNIST_C/scale/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/shear/train_images.npy')
    y_c = load('../data/MNIST_C/shear/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/shot_noise/train_images.npy')
    y_c = load('../data/MNIST_C/shot_noise/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/spatter/train_images.npy')
    y_c = load('../data/MNIST_C/spatter/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/stripe/train_images.npy')
    y_c = load('../data/MNIST_C/stripe/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/translate/train_images.npy')
    y_c = load('../data/MNIST_C/translate/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    x_c = load('../data/MNIST_C/zigzag/train_images.npy')
    y_c = load('../data/MNIST_C/zigzag/train_labels.npy')
    x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
    y_c = torch.from_numpy(y_c.reshape(60000)).long()
    for i in range(int(len(x)*MNIST_PERCENTAGE_LAYER_CKA)):
        mnist_test.append([x_c[i], y_c[i]])
    # define testing data loader
    dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=MNIST_BATCH_SIZE_LAYER_CKA, shuffle=False)
    
    # calculate the CKA distance for models using torch CKA
    cka_result = []
    for i in range(len(model_list)):
        cka_result_row = []
        for j in range(len(model_list)):
            cka = torch_cka.CKA(model_list[i], model_list[j], device=DEVICE)
            cka.compare(dataloader)
            results = cka.export()
            cka_result_row.append(results)
        cka_result.append(cka_result_row)

    return cka_result

def calculate_resnet_layer_euclidean_distance_similarity(subdataset_list, model_depth, residual, batch_norm):
    # prepare all the models
    model_list = get_resnet_model_list(subdataset_list, model_depth, residual, batch_norm)

    # get the layer weights of all models
    layerOne_list = []
    layerTwo_list = []
    layerThree_list = []
    for model in model_list:
        layerOne_list.append(model.module.conv1.weight.data)
        layerTwo_list.append(model.module.bn1.weight.data)
        layerThree_list.append(model.module.fc.weight.data)

    # calculate the euclidean distance for the first layer
    euclidean_distance_column_layerOne = []
    for i in range(len(layerOne_list)):
        euclidean_distance_row_layerOne = []
        for j in range(len(layerOne_list)):
            euclidean_distance_row_layerOne.append((layerOne_list[i] - layerOne_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column_layerOne.append(euclidean_distance_row_layerOne)
    
    # generate the distance matrix
    euclidean_distance_matrix_layerOne = np.array(euclidean_distance_column_layerOne)
    
    # calculate the euclidean distance for the second layer
    euclidean_distance_column_layerTwo = []
    for i in range(len(layerTwo_list)):
        euclidean_distance_row_layerTwo = []
        for j in range(len(layerTwo_list)):
            euclidean_distance_row_layerTwo.append((layerTwo_list[i] - layerTwo_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column_layerTwo.append(euclidean_distance_row_layerTwo)

    # generate the distance matrix
    euclidean_distance_matrix_layerTwo = np.array(euclidean_distance_column_layerTwo)

    # calculate the euclidean distance for the third layer
    euclidean_distance_column_layerThree = []
    for i in range(len(layerThree_list)):
        euclidean_distance_row_layerThree = []
        for j in range(len(layerThree_list)):
            euclidean_distance_row_layerThree.append((layerThree_list[i] - layerThree_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column_layerThree.append(euclidean_distance_row_layerThree)
    
    # generate the distance matrix
    euclidean_distance_matrix_layerThree = np.array(euclidean_distance_column_layerThree)

    # calculate the MDS of the euclidean distance matrix for layers similarity
    embedding_layerOne = MDS(n_components=1,dissimilarity='precomputed')
    X_transformed_layerOne = embedding_layerOne.fit_transform(euclidean_distance_matrix_layerOne)
    embedding_layerTwo = MDS(n_components=1,dissimilarity='precomputed')
    X_transformed_layerTwo = embedding_layerTwo.fit_transform(euclidean_distance_matrix_layerTwo)
    embedding_layerThree = MDS(n_components=1,dissimilarity='precomputed')
    X_transformed_layerThree = embedding_layerThree.fit_transform(euclidean_distance_matrix_layerThree)

    return X_transformed_layerOne, X_transformed_layerTwo, X_transformed_layerThree, euclidean_distance_matrix_layerOne, euclidean_distance_matrix_layerTwo, euclidean_distance_matrix_layerThree

def calculate_layer_cosine_similarity(dataset, subdataset_list, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # calculate the cosine similarity
    modelOne = model_list[0]
    modelTwo = model_list[1]
    modelAll = model_list[2]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarityOneTwoFirst = cos(modelOne.linear_1.weight.data, modelTwo.linear_1.weight.data)
    similarityOneTwoSecond = cos(modelOne.linear_2.weight.data, modelTwo.linear_2.weight.data)
    similarityOneAllFirst = cos(modelOne.linear_1.weight.data, modelAll.linear_1.weight.data)
    similarityOneAllSecond = cos(modelOne.linear_2.weight.data, modelAll.linear_2.weight.data)
    similarityTwoAllFirst = cos(modelTwo.linear_1.weight.data, modelAll.linear_1.weight.data)
    similarityTwoAllSecond = cos(modelTwo.linear_2.weight.data, modelAll.linear_2.weight.data)

    return similarityOneTwoFirst, similarityOneTwoSecond, similarityOneAllFirst, similarityOneAllSecond, similarityTwoAllFirst, similarityTwoAllSecond

def calculate_layer_tsne_similarity(dataset, subdataset_list, IN_DIM, OUT_DIM, STEPS):
    # create the result list
    layer1_weight_tsne_list = []
    layer2_weight_tsne_list = []

    # load the models and do the calculation
    for subdataset in subdataset_list:
        model_path = '../model/' + dataset + '/' + subdataset + '/model_final.pt'
        this_model = MLPSmall(IN_DIM, OUT_DIM)
        this_model.load_state_dict(torch.load(model_path))
        this_model.eval()
        # calculate the tsne similarity
        for i in range(STEPS):
            for j in range (STEPS):
                tsne = TSNE(n_components=2, perplexity=30, n_iter=10, verbose=1)
                layer1_weight = tsne.fit_transform(this_model.linear_1.weight.data.numpy())
                layer1_weight_tsne_list.append(layer1_weight)
                layer2_weight = tsne.fit_transform(this_model.linear_2.weight.data.numpy())
                layer2_weight_tsne_list.append(layer2_weight)
    
    return layer1_weight_tsne_list, layer2_weight_tsne_list

def calculate_model_information(dataset, subdataset_list, i, x, y, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    model = model_list[i]
    model.eval()

    # compute the accuracy, recall, precision, f1 score and confusion matrix of the perb model
    preds = model(x)
    # get the accuracy
    accuracy = Accuracy(task="multiclass", average='macro', num_classes=10)
    model_accuracy = accuracy(preds, y)
    # get the recall
    recall = Recall(task="multiclass", average='macro', num_classes=10)
    model_recall = recall(preds, y)
    # get the precision
    precision = Precision(task="multiclass", average='macro', num_classes=10)
    model_precision = precision(preds, y)
    # get the f1 score
    f1 = F1Score(task="multiclass", num_classes=10)
    model_f1 = f1(preds, y)
    # get the confusion matrix
    confusionMatrix = ConfusionMatrix(task="multiclass", num_classes=10)
    model_confusionMatrix = confusionMatrix(preds, y)

    return model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix

def calculate_torchvision_model_information(subdataset_list, model_name, i, x, y):
    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, 10)
    model = model_list[i]
    model.eval()

    # compute the accuracy, recall, precision, f1 score and confusion matrix of the perb model
    preds = model(x)
    # get the accuracy
    accuracy = Accuracy(task="multiclass", average='macro', num_classes=10)
    model_accuracy = accuracy(preds, y)
    # get the recall
    recall = Recall(task="multiclass", average='macro', num_classes=10)
    model_recall = recall(preds, y)
    # get the precision
    precision = Precision(task="multiclass", average='macro', num_classes=10)
    model_precision = precision(preds, y)
    # get the f1 score
    f1 = F1Score(task="multiclass", num_classes=10)
    model_f1 = f1(preds, y)
    # get the confusion matrix
    confusionMatrix = ConfusionMatrix(task="multiclass", num_classes=10)
    model_confusionMatrix = confusionMatrix(preds, y)

    return model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix

def calculate_hessian_loss_landscape(dataset, subdataset_list, i, criterion, x, y, IN_DIM, OUT_DIM, STEPS, START, END):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    model = model_list[i]
    model_perb = copy.deepcopy(model)
    model_perb.eval()

    # store the accuracy, recall, precision, f1 score, confusion matrix
    model_accuracy = []
    model_recall = []
    model_precision = []
    model_f1 = []
    model_confusionMatrix = []

    if FLAG == True:
        model.cuda()
        model_perb.cuda()
        x = x.cuda()
        y = y.cuda()

    # calculate hessian contour and save model information
    hessian_comp = hessian(model, criterion, data=(x, y), cuda=FLAG)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    lams = np.linspace(START, END, STEPS).astype(np.float32)
    loss_list = []
    model_list = []
    for lam in lams:
        model_perb_fir = get_params(model, model_perb, top_eigenvector[0], lam)
        loss_list_sec = []
        model_list_sec = []
        for lam_sec in lams:
            model_perb_sec = copy.deepcopy(model_perb_fir)
            model_perb_sec = get_params(model_perb_fir, model_perb_sec, top_eigenvector[1], lam_sec)
            loss_list_sec.append(criterion(model_perb_sec(x), y).item())
            model_to_be_saved = copy.deepcopy(model_perb_sec).eval()
            # compute the accuracy, recall, precision, f1 score and confusion matrix of the perb model
            preds = model_to_be_saved(x)
            # get the accuracy
            accuracy = Accuracy(subset_accuracy=True)
            model_accuracy.append(accuracy(preds, y))
            # get the recall
            recall = Recall(average='macro', num_classes=10)
            model_recall.append(recall(preds, y))
            # get the precision
            precision = Precision(average='macro', num_classes=10)
            model_precision.append(precision(preds, y))
            # get the f1 score
            f1 = F1Score(num_classes=10)
            model_f1.append(f1(preds, y))
            # get the confusion matrix
            confusionMatrix = ConfusionMatrix(num_classes=10)
            model_confusionMatrix.append(confusionMatrix(preds, y))
            model_list_sec.append(model_to_be_saved)
        loss_list.append(loss_list_sec)
        model_list.append(model_list_sec)

    # generate the result string
    result_array = np.array(loss_list)
    result_string = np.array2string(result_array, precision=4, separator=',', suppress_small=True)
    result_string = ""
    for i in range(len(loss_list)):
        for j in range (len(loss_list[i])):
            if (i != STEPS-1 or j != STEPS-1):
                result_string += str(loss_list[i][j]) + ","
            else:
                result_string += str(loss_list[i][j])
    
    return top_eigenvalues, result_string, result_array, model_list, model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix

def calculate_resnet_hessian_loss_landscape(subdataset_list, model_depth, residual, batch_norm, i, criterion, x, y, STEPS, START, END):
    # prepare all the models
    model_list = get_resnet_model_list(subdataset_list, model_depth, residual, batch_norm)
    model = model_list[i]
    model_perb = copy.deepcopy(model)
    model_perb.eval()

    # store the accuracy, recall, precision, f1 score, confusion matrix
    model_accuracy = []
    model_recall = []
    model_precision = []
    model_f1 = []
    model_confusionMatrix = []

    if FLAG == True:
        model.cuda()
        model_perb.cuda()
        x = x.cuda()
        y = y.cuda()
    
    # calculate hessian contour and save model information
    hessian_comp = hessian(model, criterion, data=(x, y), cuda=FLAG)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    lams = np.linspace(START, END, STEPS).astype(np.float32)
    loss_list = []
    model_list = []
    for lam in lams:
        model_perb_fir = get_params(model, model_perb, top_eigenvector[0], lam)
        loss_list_sec = []
        model_list_sec = []
        for lam_sec in lams:
            model_perb_sec = copy.deepcopy(model_perb_fir)
            model_perb_sec = get_params(model_perb_fir, model_perb_sec, top_eigenvector[1], lam_sec)
            loss_list_sec.append(criterion(model_perb_sec(x), y).item())
            model_to_be_saved = copy.deepcopy(model_perb_sec).eval()
            # compute the accuracy, recall, precision, f1 score and confusion matrix of the perb model
            preds = model_to_be_saved(x)
            # get the accuracy
            accuracy = Accuracy(subset_accuracy=True)
            model_accuracy.append(accuracy(preds, y))
            # get the recall
            recall = Recall(average='macro', num_classes=10)
            model_recall.append(recall(preds, y))
            # get the precision
            precision = Precision(average='macro', num_classes=10)
            model_precision.append(precision(preds, y))
            # get the f1 score
            f1 = F1Score(num_classes=10)
            model_f1.append(f1(preds, y))
            # get the confusion matrix
            confusionMatrix = ConfusionMatrix(num_classes=10)
            model_confusionMatrix.append(confusionMatrix(preds, y))
            model_list_sec.append(model_to_be_saved)
        loss_list.append(loss_list_sec)
        model_list.append(model_list_sec)

    # generate the result string
    result_array = np.array(loss_list)
    result_string = np.array2string(result_array, precision=4, separator=',', suppress_small=True)
    result_string = ""
    for i in range(len(loss_list)):
        for j in range (len(loss_list[i])):
            if (i != STEPS-1 or j != STEPS-1):
                result_string += str(loss_list[i][j]) + ","
            else:
                result_string += str(loss_list[i][j])
    
    return top_eigenvalues, result_string, model_list, model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix

def calculate_loss_landscapes_random_projection(dataset, subdataset_list, criterion, x, y, IN_DIM, OUT_DIM, STEPS):
    # prepare the result list
    loss_data_fin_list = []
    max_loss_value_list = []
    min_loss_value_list = []
    model_info_list = []
    
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    
    # calculate the first original sub-dataset and get one random projection
    metric = loss_landscapes.metrics.Loss(criterion, x, y)
    
    # calculate the loss landscape in 2 dimensions for the first model
    # loss_data_fin, dir_one, dir_two = loss_landscapes.random_plane(model_list[0], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
    # loss_data_fin, model_info, dir_one, dir_two = loss_landscapes_main.random_plane(model_list[0], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
    # loss_data_fin_list.append(loss_data_fin)
    # model_info_list.append(model_info)

    # # first array corresponds to which row, and the latter corresponds to which column
    # max_loss = np.where(loss_data_fin == np.max(loss_data_fin))
    # max_loss_value_list.append(max_loss[0][0])
    # max_loss_value_list.append(max_loss[1][0])
    # min_loss = np.where(loss_data_fin == np.min(loss_data_fin))
    # min_loss_value_list.append(min_loss[0][0])
    # min_loss_value_list.append(min_loss[1][0])
    
    # calculate the loss landscape in 2 dimensions for the rest of the models
    for i in range(len(model_list)):
        # calculate the loss landscape in 2 dimensions for the model
        loss_data_fin_this, model_info_this, dir_one, dir_two = loss_landscapes_main.random_plane(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_this)
        model_info_list.append(model_info_this)

        # first array corresponds to which row, and the latter corresponds to which colresnet50umn
        max_loss_this = np.where(loss_data_fin_this == np.max(loss_data_fin_this))
        max_loss_value_list.append(max_loss_this[0][0])
        max_loss_value_list.append(max_loss_this[1][0])
        min_loss_this = np.where(loss_data_fin_this == np.min(loss_data_fin_this))
        min_loss_value_list.append(min_loss_this[0][0])
        min_loss_value_list.append(min_loss_this[1][0])
        
    return loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list

def calculate_resnet_loss_landscapes_random_projection(subdataset_list, model_depth, residual, batch_norm, i, criterion, x, y, STEPS):
    # prepare the result list
    loss_data_fin_list = []
        
    # prepare all the models
    model_list = get_resnet_model_list(subdataset_list, model_depth, residual, batch_norm)
    
    # calculate the first original sub-dataset and get one random projection
    metric = loss_landscapes.metrics.Loss(criterion, x, y)
    
    # calculate the loss landscape in 2 dimensions for the first model
    loss_data_fin, dir_one, dir_two = loss_landscapes.random_plane(model_list[0], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
    loss_data_fin_list.append(loss_data_fin)

    # calculate the loss landscape in 2 dimensions for the rest of the models
    for i in range(1, len(model_list)):
        # calculate the loss landscape in 2 dimensions for the model
        loss_data_fin_this = loss_landscapes.random_plane_given_plane(model_list[i], metric, dir_one, dir_two, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_this)

    return loss_data_fin_list

def calculate_3d_loss_landscapes_random_projection(dataset, subdataset_list, criterion, x, y, IN_DIM, OUT_DIM, STEPS):
    # prepare the result list
    loss_data_fin_list = []

    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    
    # calculate the first original sub-dataset and get one random projection
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    # # compute loss landscape 3D data for the first model
    # loss_data_fin_3d, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_list[0], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
    # loss_data_fin_list.append(loss_data_fin_3d)

    # calculate the loss landscape in 3 dimensions for the rest of the models
    for i in range(len(model_list)):
        # compute loss landscape 3D data for the rest of the models
        loss_data_fin_3d, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_3d)
    
    return loss_data_fin_list

def get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM):
    # load all the models and store them into a list
    model_list = []
    for subdataset in subdataset_list:
        model_path = '../model/' + dataset + '/' + subdataset + '/model_final.pt'
        this_model = MLPSmall(IN_DIM, OUT_DIM)
        this_model.load_state_dict(torch.load(model_path))
        this_model.eval()
        model_list.append(this_model)
    return model_list

def store_into_database(client, database_name, collection_name, record):
    if(functions.empty(client,database_name,collection_name) == True):
        print("The " + collection_name + " collection does not exist.")
        print("Create the new collection named " + collection_name + ".")
        functions.create(client,database_name,collection_name)

    record_id = functions.insert(client,database_name,collection_name, record)
    return record_id

def store_into_file(collection_name, record, number_of_indent, model_name):
    # Serializing json
    json_object = json.dumps(record, indent=number_of_indent)
    if model_name == 'RESNET18':
        with open('resnet18_result_files/' + collection_name + '.json', 'w') as outfile:
            outfile.write(json_object)
    elif model_name == 'CNN':
        with open('cnn_result_files/' + collection_name + '.json', 'w') as outfile:
            outfile.write(json_object)
    
    return True

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,inplanes,planes,residual_not,batch_norm_not,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.batch_norm_not:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes,planes,residual_not,batch_norm_not,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        if self.batch_norm_not:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batch_norm_not:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual

        out = self.relu(out)

        return out

ALPHA_ = 1

class ResNet(nn.Module):

    def __init__(self,depth,residual_not=True,batch_norm_not=True,base_channel=16,num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        # block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock
        self.base_channel = int(base_channel)
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.inplanes = self.base_channel * ALPHA_
        self.conv1 = nn.Conv2d(3,self.base_channel * ALPHA_,kernel_size=3,padding=1,bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(self.base_channel * ALPHA_)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,self.base_channel * ALPHA_,n,self.residual_not,self.batch_norm_not)
        self.layer2 = self._make_layer(block,self.base_channel * 2 * ALPHA_,n,self.residual_not,self.batch_norm_not,stride=2)
        self.layer3 = self._make_layer(block,self.base_channel * 4 * ALPHA_,n,self.residual_not,self.batch_norm_not,stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.base_channel * 4 * ALPHA_ * block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,block,planes,blocks,residual_not,batch_norm_not,stride=1):
        downsample = None
        if (stride != 1 or
                self.inplanes != planes * block.expansion) and (residual_not):
            if batch_norm_not:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,planes * block.expansion,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,planes * block.expansion,kernel_size=1,stride=stride,bias=False),)

        layers = nn.ModuleList()
        layers.append(
            block(self.inplanes, planes, residual_not, batch_norm_not, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, residual_not, batch_norm_not))

        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        output_list = []
        x = self.conv1(x)
        if self.batch_norm_not:
            x = self.bn1(x)
        x = self.relu(x)  # 32x32
        output_list.append(x.reshape(x.size(0), -1))

        for layer in self.layer1:
            x = layer(x)  # 32x32
            output_list.append(x.reshape(x.size(0), -1))
        for layer in self.layer2:
            x = layer(x)  # 16x16
            output_list.append(x.reshape(x.size(0), -1))
        for layer in self.layer3:
            x = layer(x)  # 8x8
            output_list.append(x.reshape(x.size(0), -1))

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        output_list.append(x.reshape(x.size(0), -1))

        # return output_list, x
        return x

def resnet(**kwargs):
    # Constructs a ResNet model.
    return ResNet(**kwargs)

def get_resnet_model_list(subdataset_list, model_depth, residual, batch_norm):
    # load all the models and store them into a list
    model_list = []
    for subdataset in subdataset_list:
        model_path = '../model/CIFAR10/RESNET/resnet' + str(model_depth) + '_cifar10_' + subdataset + '_model_final.pt'
        this_model = resnet(num_classes=10, depth=model_depth, residual_not=residual, batch_norm_not=batch_norm)
        this_model = torch.nn.DataParallel(this_model)
        this_model.load_state_dict(torch.load(model_path))
        this_model.eval()
        model_list.append(this_model)
    return model_list

def calculate_hessian_loss_landscape_plot(dataset, subdataset_list, criterion, x, y, IN_DIM, OUT_DIM, STEPS):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)

    # prepare the result list
    loss_data_fin_list = []
    max_loss_value_list = []
    min_loss_value_list = []
    top_eigenvalues_list = []

    for i in range(len(model_list)):
        # calculate the hessian loss landscape in 2 dimensions for the first of the models
        metric = loss_landscapes.metrics.Loss(criterion, x, y)
        hessian_comp = hessian(model_list[i], criterion, data=(x, y), cuda=False)
        
        # calculate loss values for 2d loss contour
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
        dir_one_hessian = loss_landscapes.ModelParameters(top_eigenvector[0])
        dir_two_hessian = loss_landscapes.ModelParameters(top_eigenvector[1])
        loss_data_fin_hessian, dir_one_hessian, dir_two_hessian = loss_landscapes.random_plane_hessian(model_list[i], metric, dir_one_hessian, dir_two_hessian, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_hessian)
        top_eigenvalues_list.append(top_eigenvalues)
        
        # first array corresponds to which row, and the latter corresponds to which column
        max_loss_this = np.where(loss_data_fin_hessian == np.max(loss_data_fin_hessian))
        max_loss_value_list.append(max_loss_this[0][0])
        max_loss_value_list.append(max_loss_this[1][0])
        min_loss_this = np.where(loss_data_fin_hessian == np.min(loss_data_fin_hessian))
        min_loss_value_list.append(min_loss_this[0][0])
        min_loss_value_list.append(min_loss_this[1][0])

    return loss_data_fin_list, max_loss_value_list, min_loss_value_list, top_eigenvalues_list

def get_torchvision_original_model(name, num_classes):
    model = None
    if name == 'RESNET18':
        model = torchvision.models.resnet18(weights = None)
        model.fc = nn.Linear(512, num_classes)
    elif name == 'RESNET50':
        model = torchvision.models.resnet50(weights = None)
        model.fc = nn.Linear(2048, num_classes)
    elif name == 'VGG16':
        model = torchvision.models.vgg16(weights = None)
    return model

def get_data_cifar10(CIFAR10_BATCH_SIZE, batch_size_test):

    train_set = torchvision.datasets.CIFAR10('../data/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
    ]))

    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=CIFAR10_BATCH_SIZE,drop_last=False)
    
    test_set = torchvision.datasets.CIFAR10('../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                               
                             ]))

    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test)
    return train_subset_loader, test_subset_loader

def get_data_cifar100(CIFAR10_BATCH_SIZE, batch_size_test):

    train_set = torchvision.datasets.CIFAR100('../data/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
    ]))

    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=CIFAR10_BATCH_SIZE,drop_last=False)
    
    test_set = torchvision.datasets.CIFAR100('../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))

    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test)
    return train_subset_loader, test_subset_loader

def get_torchvision_model_list(subdataset_list, model_name, num_classes):
    # load all the models and store them into a list
    model_list = []
    for subdataset in subdataset_list:
        threshold = 'threshold_{100}'
        if subdataset == 'all':
            threshold = 'threshold_{100}'
        elif subdataset == 'original':
            threshold = 'threshold_{00}'
        else:
            threshold = subdataset
        model_path = '../model/CIFAR10/' + model_name + '/' + model_name + '_model_' + threshold + '.pt'
        this_model = get_torchvision_original_model(model_name, num_classes)
        this_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        this_model.eval()
        model_list.append(this_model)
    return model_list

def calculate_torchvision_model_similarity_global_structure(subdataset_list, model_name, num_classes):
    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)

    # join the tensors of all subdatasets
    modelTensor_list = []
    if model_name == 'RESNET18':
        for model in model_list:
            modelTensor_list.append(torch.cat((model.conv1.weight.data.reshape(-1),model.bn1.weight.data.reshape(-1),model.fc.weight.data.reshape(-1))))
    elif model_name == 'RESNET50':
        for model in model_list:
            modelTensor_list.append(torch.cat((model.conv1.weight.data.reshape(-1),model.bn1.weight.data.reshape(-1),model.fc.weight.data.reshape(-1))))
    elif model_name == 'VGG16':
        for model in model_list:
            modelTensor_list.append(torch.cat((model.features[0].weight.data.reshape(-1),model.classifier[0].weight.data.reshape(-1))))
    
    # calculate the euclidean distance between models
    euclidean_distance_column = []
    for i in range(len(model_list)):
        euclidean_distance_row = []
        for j in range(len(model_list)):
            euclidean_distance_row.append((modelTensor_list[i] - modelTensor_list[j]).pow(2).sum().sqrt())
        euclidean_distance_column.append(euclidean_distance_row)

    # generate the distance matrix
    euclidean_distance_matrix = np.array(euclidean_distance_column)
    
    # calculate the MDS of the euclidean distance matrix for models similarity
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(euclidean_distance_matrix)

    # calculate the global structure of the models based on the MDS
    condensed_distance_matrix = pdist(X_transformed)
    Z = hierarchy.linkage(condensed_distance_matrix, 'single')
    T = to_tree(Z, rd=False)
    figure  = to_newick(T, ascii_lowercase)
    
    return X_transformed, figure

def calculate_torchvision_model_cka_similarity_global_structure(subdataset_list, model_name, num_classes, reshape_x, reshape_y):
    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)

    # join the tensors of all subdatasets
    modelTensor_list = []
    if model_name == 'RESNET18':
        for model in model_list:
            modelTensor_list.append(torch.cat((model.conv1.weight.data.reshape(-1),model.bn1.weight.data.reshape(-1),model.fc.weight.data.reshape(-1))))
    elif model_name == 'RESNET50':
        for model in model_list:
            modelTensor_list.append(torch.cat((model.conv1.weight.data.reshape(-1),model.bn1.weight.data.reshape(-1),model.fc.weight.data.reshape(-1))))
    elif model_name == 'VGG16':
        for model in model_list:
            modelTensor_list.append(torch.cat((model.features[0].weight.data.reshape(-1),model.classifier[0].weight.data.reshape(-1))))
    
    # initialize the CKA class
    np_cka = CKA()
    
    # calculate the linear CKA similarity between models
    linear_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        linear_cka_similarity_row = []
        for j in range(len(model_list)):
            linear_cka_similarity_row.append(np_cka.linear_CKA(modelTensor_list[i].numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].numpy().reshape((reshape_x, reshape_y))))
        linear_cka_similarity_column.append(linear_cka_similarity_row)
    
    kernel_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        kernel_cka_similarity_row = []
        for j in range(len(model_list)):
            kernel_cka_similarity_row.append(np_cka.kernel_CKA(modelTensor_list[i].numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].numpy().reshape((reshape_x, reshape_y))))
        kernel_cka_similarity_column.append(kernel_cka_similarity_row)
    
    # generate the distance matrix for linear CKA
    linear_cka_matrix = 1 - np.array(linear_cka_similarity_column)
    
    # generate the distance matrix for RBF kernel CKA
    kernel_cka_matrix = 1 - np.array(kernel_cka_similarity_column)
    
    # calculate the MDS of the linear CKA distance matrix for models similarity
    linear_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    linear_cka_embedding = linear_cka_mds.fit_transform(linear_cka_matrix)

    # calculate the global structure of the models based on the linear CKA MDS
    linear_cka_condensed_distance_matrix = pdist(linear_cka_embedding)
    linear_cka_Z = hierarchy.linkage(linear_cka_condensed_distance_matrix, 'single')
    linear_cka_T = to_tree(linear_cka_Z, rd=False)
    linear_cka_figure  = to_newick(linear_cka_T, ascii_lowercase)

    # calculate the MDS of the RBF kernel CKA distance matrix for models similarity
    kernel_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    kernel_cka_embedding = kernel_cka_mds.fit_transform(kernel_cka_matrix)

    # calculate the global structure of the models based on the RBF kernel CKA MDS
    kernel_cka_condensed_distance_matrix = pdist(kernel_cka_embedding)
    kernel_cka_Z = hierarchy.linkage(kernel_cka_condensed_distance_matrix, 'single')
    kernel_cka_T = to_tree(kernel_cka_Z, rd=False)
    kernel_cka_figure  = to_newick(kernel_cka_T, ascii_lowercase)

    return linear_cka_embedding, kernel_cka_embedding, linear_cka_figure, kernel_cka_figure

def calculate_torchvision_model_layer_euclidean_distance_similarity(subdataset_list, model_name, num_classes):
    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)

    all_model_layer_list = []
    # get the layer weights of all models
    for model in model_list:
        current_model_layer_list = []
        for name, param in model.named_parameters():
            if 'conv' in name and 'weight' in name:
                current_model_layer_list.append(param.data)
        all_model_layer_list.append(current_model_layer_list)
    
    # calculate the euclidean distance for layers
    euclidean_distance_column_all_layer = []
    for k in range(len(all_model_layer_list[0])):
        euclidean_distance_column_current_layer = []
        for i in range(len(all_model_layer_list)):
            euclidean_distance_row_current_layer = []
            for j in range(len(all_model_layer_list)):
                euclidean_distance_row_current_layer.append((all_model_layer_list[i][k] - all_model_layer_list[j][k]).pow(2).sum().sqrt())
            euclidean_distance_column_current_layer.append(euclidean_distance_row_current_layer)
        euclidean_distance_column_all_layer.append(euclidean_distance_column_current_layer)
    
    # generate the distance matrix for layers
    euclidean_distance_matrix_all_layer = []
    for i in range(len(euclidean_distance_column_all_layer)):
        euclidean_distance_matrix_all_layer.append(np.array(euclidean_distance_column_all_layer[i]))
    
    # calculate the MDS of the euclidean distance matrix for layers similarity
    embedding_layer = MDS(n_components=1,dissimilarity='precomputed')
    X_transformed_layer = []
    for i in range(len(euclidean_distance_matrix_all_layer)):
        X_transformed_layer.append(embedding_layer.fit_transform(euclidean_distance_matrix_all_layer[i]))
    
    return X_transformed_layer, euclidean_distance_matrix_all_layer

def calculate_torchvision_model_resnet_hessian_loss_landscape(subdataset_list, model_name,num_classes, i, criterion, x, y, STEPS, START, END):
    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)
    model = model_list[i]
    model_perb = copy.deepcopy(model)
    model_perb.eval()

    # store the accuracy, recall, precision, f1 score, confusion matrix
    model_accuracy = []
    model_recall = []
    model_precision = []
    model_f1 = []
    model_confusionMatrix = []

    if FLAG == True:
        model = model.cuda()
        model_perb = model_perb.cuda()
        criterion = criterion.cuda()
        x = x.cuda()
        y = y.cuda()
    
    # calculate hessian contour and save model information
    hessian_comp = hessian(model, criterion, data=(x, y), cuda=FLAG)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    lams = np.linspace(START, END, STEPS).astype(np.float32)
    loss_list = []
    model_list = []
    for lam in lams:
        model_perb_fir = get_params(model, model_perb, top_eigenvector[0], lam)
        loss_list_sec = []
        model_list_sec = []
        for lam_sec in lams:
            model_perb_sec = copy.deepcopy(model_perb_fir)
            model_perb_sec = get_params(model_perb_fir, model_perb_sec, top_eigenvector[1], lam_sec)
            loss_list_sec.append(criterion(model_perb_sec(x), y).item())
            model_to_be_saved = copy.deepcopy(model_perb_sec).eval()
            # compute the accuracy, recall, precision, f1 score and confusion matrix of the perb model
            preds = model_to_be_saved(x)
            # get the accuracy
            accuracy = Accuracy(subset_accuracy=True)
            model_accuracy.append(accuracy(preds, y))
            # get the recall
            recall = Recall(average='macro', num_classes=10)
            model_recall.append(recall(preds, y))
            # get the precision
            precision = Precision(average='macro', num_classes=10)
            model_precision.append(precision(preds, y))
            # get the f1 score
            f1 = F1Score(num_classes=10)
            model_f1.append(f1(preds, y))
            # get the confusion matrix
            confusionMatrix = ConfusionMatrix(num_classes=10)
            model_confusionMatrix.append(confusionMatrix(preds, y))
            model_list_sec.append(model_to_be_saved)
        loss_list.append(loss_list_sec)
        model_list.append(model_list_sec)

    # generate the result string
    result_array = np.array(loss_list)
    result_string = np.array2string(result_array, precision=4, separator=',', suppress_small=True)
    result_string = ""
    for i in range(len(loss_list)):
        for j in range (len(loss_list[i])):
            if (i != STEPS-1 or j != STEPS-1):
                result_string += str(loss_list[i][j]) + ","
            else:
                result_string += str(loss_list[i][j])
    
    return top_eigenvalues, result_string, result_array, model_list, model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix

def calculate_torchvision_model_loss_landscapes_random_projection(subdataset_list, model_name, num_classes, criterion, x, y, STEPS):
    # prepare the result list
    loss_data_fin_list = []
    model_info_list = []
    max_loss_value_list = []
    min_loss_value_list = []

    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)

    # calculate the first original sub-dataset and get one random projection
    metric = sl_metrics.LossTorchvision(criterion, x, y)
    
    # calculate the loss landscape in 2 dimensions for the first model
    # loss_data_fin, dir_one, dir_two = loss_landscapes.random_plane(model_list[0], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
    # loss_data_fin, model_info, dir_one, dir_two = loss_landscapes_main.random_plane(model_list[0], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
    # loss_data_fin_array = np.array(loss_data_fin)
    # loss_data_fin_array = np.where(np.isnan(loss_data_fin_array), ma.array(loss_data_fin_array, mask=np.isnan(loss_data_fin_array)).max(axis=1), loss_data_fin_array)
    # loss_data_fin = loss_data_fin_array.tolist()
    # loss_data_fin_list.append(loss_data_fin)
    # model_info_list.append(model_info)

    # # first array corresponds to which row, and the latter corresponds to which column
    # max_loss = np.where(loss_data_fin == np.max(loss_data_fin))
    # max_loss_value_list.append(max_loss[0][0])
    # max_loss_value_list.append(max_loss[1][0])
    # min_loss = np.where(loss_data_fin == np.min(loss_data_fin))
    # min_loss_value_list.append(min_loss[0][0])
    # min_loss_value_list.append(min_loss[1][0])

    # calculate the loss landscape in 2 dimensions for the rest of the models
    for i in range(len(model_list)):
        # calculate the loss landscape in 2 dimensions for the model
        loss_data_fin_this, model_info_this, dir_one, dir_two = loss_landscapes_main.random_plane_rmbn2(model_list[i], metric, 0.1, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_this)
        model_info_list.append(model_info_this)

        # first array corresponds to which row, and the latter corresponds to which colresnet50umn
        max_loss_this = np.where(loss_data_fin_this == np.max(loss_data_fin_this))
        max_loss_value_list.append(max_loss_this[0][0])
        max_loss_value_list.append(max_loss_this[1][0])
        min_loss_this = np.where(loss_data_fin_this == np.min(loss_data_fin_this))
        min_loss_value_list.append(min_loss_this[0][0])
        min_loss_value_list.append(min_loss_this[1][0])
        
    return loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list

def calculate_torchvision_model_3d_loss_landscapes_random_projection(subdataset_list, model_name, num_classes, criterion, x, y, STEPS):
    # prepare the result list
    loss_data_fin_list = []

    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)
    
    # calculate the first original sub-dataset and get one random projection
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    # # compute loss landscape 3D data for the first model
    # loss_data_fin_3d, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_list[0], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
    # loss_data_fin_list.append(loss_data_fin_3d)

    # calculate the loss landscape in 3 dimensions for all models
    for i in range(len(model_list)):
        # compute loss landscape 3D data for the rest of the models
        loss_data_fin_3d, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_3d)
    
    return loss_data_fin_list

def calculate_torchvision_model_training_3d_loss_landscapes_random_projection(subdataset_list, model_name, num_classes, criterion, STEPS):
    # prepare the result list
    loss_data_fin_list = []

    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)
    
    # calculate the loss landscape in 3 dimensions for all models
    for i in range(len(model_list)):
        # prepare the training data based on the model
        x, y = get_cifar10_for_one_model(subdataset_list[i])

        # prepare the metric
        metric = loss_landscapes.metrics.Loss(criterion, x, y)

        for i in range(len(model_list)):
            # compute loss landscape 3D data for the rest of the models
            loss_data_fin_3d, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
            loss_data_fin_list.append(loss_data_fin_3d)

    return loss_data_fin_list

def get_selected_model_list_from_loss_landscapes(loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list, threshold_percent, num_models):
    selected_model_list_all = []
    for i in range(len(loss_data_fin_list)):
        loss_data_fin = loss_data_fin_list[i]
        model_info = model_info_list[i]
        max_loss_value = max_loss_value_list[i]
        min_loss_value = min_loss_value_list[i]
        whole_gap = max_loss_value - min_loss_value
        threshold  = min_loss_value + whole_gap * threshold_percent
        selected_model_list = []
        for j in range(len(loss_data_fin)):
            for k in range(len(loss_data_fin[j])):
                if loss_data_fin[j][k] <= threshold and len(selected_model_list) < num_models:
                    selected_model_list.append(model_info[j][k])
        selected_model_list_all.append(selected_model_list)
    return selected_model_list_all

def get_selected_center_model_list_from_loss_landscapes(model_info_list, num_models_x, num_models_y, LOSS_STEPS):
    # create the list to store all the selected models
    selected_model_list_all = []

    for k in range(len(model_info_list)):
        # unpack the model_info_list in the first level
        model_set_one_column = model_info_list[k]
        model_set_one = []

        print("Length of model_set_one_column: ", len(model_set_one_column))

        # further unpack the model_info_list in the second level
        for i in range(len(model_set_one_column)):
            for j in range(len(model_set_one_column[i])):
                model_set_one.append(model_set_one_column[i][j])
        
        # create the list to store the selected models
        selected_model_one_list = []

        print("Length of model_set_one: ", len(model_set_one))
        
        # select the models
        for i in range(math.floor(LOSS_STEPS/2)-math.floor(num_models_x/2)-1, math.floor(LOSS_STEPS/2)-math.floor(num_models_x/2)-1+num_models_x):
            for j in range(math.floor(LOSS_STEPS/2)-math.floor(num_models_y/2)-1, math.floor(LOSS_STEPS/2)-math.floor(num_models_y/2)-1+num_models_y):
                model_info_one = model_set_one[LOSS_STEPS*j+i]
                selected_model_one_list.append(model_info_one)
        
        # pack the selected models into one list
        selected_model_list_all.append(selected_model_one_list)
    
    return selected_model_list_all

def calculate_detailed_similarity_from_loss_landscapes_models(selected_model_list_all, i, j):
    selected_models_one = copy.deepcopy(selected_model_list_all[i])
    selected_models_two = copy.deepcopy(selected_model_list_all[j])
    this_whole_model_list = []
    for m in range(len(selected_models_one)):
        this_whole_model_list.append(selected_models_one[m])
    for n in range(len(selected_models_two)):
        this_whole_model_list.append(selected_models_two[n])
        
    # calculate the euclidean distance between models
    euclidean_distance_column = []
    for p in range(len(this_whole_model_list)):
        euclidean_distance_row = []
        for q in range(len(this_whole_model_list)):
            euclidean_distance_row.append(torch.tensor(np.array(this_whole_model_list[p]) - np.array(this_whole_model_list[q])).pow(2).sum().sqrt())
        euclidean_distance_column.append(euclidean_distance_row)

    # generate the distance matrix
    euclidean_distance_matrix = np.array(euclidean_distance_column)
    # calculate the MDS of the euclidean distance matrix for models similarity
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(euclidean_distance_matrix)
    
    return X_transformed

def transfer_model_parameters_into_models(selected_model_list, IN_DIM, OUT_DIM):
    selected_all_model_list = []
    for i in range(len(selected_model_list)):
        # choose two model sets
        model_set_one = selected_model_list[i]
        # get models from the model
        model_set = []
        for j in range(len(model_set_one)):
            model_one_in_set_one = np.array(model_set_one[j]).reshape(25450,1)
            linear1_weight_array = model_one_in_set_one[slice(int(0), int(784*32))]
            linear2_weight_array = model_one_in_set_one[slice(int(784*32+32), int(784*32+32*11))]
            linear1_bias_array = model_one_in_set_one[slice(int(784*32), int(784*32+32))]
            linear2_bias_array = model_one_in_set_one[slice(int(784*32+32*11), int(784*32+32*11+10))]
            linear1_weight = torch.nn.Parameter(torch.from_numpy(linear1_weight_array.reshape(32,784)))
            linear2_weight = torch.nn.Parameter(torch.from_numpy(linear2_weight_array.reshape(10,32)))
            linear1_bias = torch.nn.Parameter(torch.from_numpy(linear1_bias_array.reshape(32)))
            linear2_bias = torch.nn.Parameter(torch.from_numpy(linear2_bias_array.reshape(10)))
            # put the linear weight into the model
            modify_model = MLPSmallModify(IN_DIM, OUT_DIM, linear1_weight, linear2_weight, linear1_bias, linear2_bias)
            modify_model.eval()
            model_set.append(modify_model)
        selected_all_model_list.append(model_set)
    return selected_all_model_list

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def calculate_prediction_distribution(model_set_one, model_set_two, distribution_test_loader, MAX_NUM):
    # prepare the result set
    correct_correct = set()
    correct_wrong = set()
    wrong_correct = set()
    wrong_wrong = set()

    # create the iterator for the testing dataset
    distribution_test_loader_iter = iter(distribution_test_loader)

    print("Length of the testing dataset: ", len(distribution_test_loader))

    # calculate the prediction distribution result for each testing image
    for i in range(len(distribution_test_loader)):
    # for i in range(1000):
        # get one testing image
        # try:
        #     while True:
        #         distribution_x, distribution_y = distribution_test_loader_iter.__next__()
        # except StopIteration:
        #     pass
        distribution_x, distribution_y = distribution_test_loader_iter.__next__()
        # predict all the results of the first model set using voting
        result_one_list = []
        for j in range(len(model_set_one)):
            model = model_set_one[j]
            model.eval()
            prediction = model(distribution_x).reshape(1,10)
            result = torch.argmax(prediction, dim=1).item()
            result_one_list.append(result)
        # calculate the voting result for the first model set
        voting_result_one = most_frequent(result_one_list)
        # predict all the results of the second model set using voting
        result_two_list = []
        for k in range(len(model_set_two)):
            model = model_set_two[k]
            model.eval()
            prediction = model(distribution_x).reshape(1,10)
            result = torch.argmax(prediction, dim=1).item()
            result_two_list.append(result)
        # calculate the voting result for the second model set
        voting_result_two = most_frequent(result_two_list)
        # get the correct label
        label_y = distribution_y.item()
        # print("The correct label is: ", label_y)
        # print("The voting result for the first model set is: ", voting_result_one)
        # print("The voting result for the second model set is: ", voting_result_two)

        # calculate the prediction distribution result for one testing image
        if voting_result_one == voting_result_two:
            if voting_result_one == label_y:
                if len(correct_correct) < MAX_NUM:
                    correct_correct.add(i)
            else:
                if len(wrong_wrong) < MAX_NUM:
                    wrong_wrong.add(i)
        else:
            if voting_result_one == label_y:
                if len(correct_wrong) < MAX_NUM:
                    correct_wrong.add(i)
            elif voting_result_two == label_y:
                if len(wrong_correct) < MAX_NUM:
                    wrong_correct.add(i)
            else:
                if len(wrong_wrong) < MAX_NUM:
                    wrong_wrong.add(i)
        
        # check if the result sets are full
        if len(correct_correct) >= MAX_NUM and len(correct_wrong) >= MAX_NUM and len(wrong_correct) >= MAX_NUM and len(wrong_wrong) >= MAX_NUM:
            break

    # return the result sets as lists
    return list(correct_correct), list(correct_wrong), list(wrong_correct), list(wrong_wrong)

def calculate_torchvision_model_prediction_distribution(model_one, model_two, distribution_test_loader, MAX_NUM):
    # prepare the result set
    correct_correct = set()
    correct_wrong = set()
    wrong_correct = set()
    wrong_wrong = set()

    # predict each testing image one by one
    model_one.eval()
    model_two.eval()

    # create the iterator for the testing dataset
    distribution_test_loader_iter = iter(distribution_test_loader)

    print("Length of the testing dataset: ", len(distribution_test_loader))

    # calculate the prediction distribution result for each testing image
    for i in range(len(distribution_test_loader)):
    # for i in range(10000):
        # get one testing image
        try:
            distribution_x, distribution_y = distribution_test_loader_iter.__next__()
        except StopIteration:
            distribution_test_loader_iter = iter(distribution_test_loader)
            distribution_x, distribution_y = next(distribution_test_loader_iter)
        # prediction by the first model
        distribution_x = distribution_x.reshape(1, 3, 32, 32)
        prediction_one = model_one(distribution_x)
        result_one = torch.argmax(prediction_one, dim=1).item()
        # prediction by the second model
        prediction_two = model_two(distribution_x)
        result_two = torch.argmax(prediction_two, dim=1).item()
        # get the correct label
        label_y = distribution_y.item()
        # print("The correct label is: ", label_y)
        # print("The prediction result for the first model is: ", result_one)
        # print("The prediction result for the second model is: ", result_two)

        # calculate the prediction distribution result for one testing image
        if result_one == result_two:
            if result_one == label_y:
                if len(correct_correct) < MAX_NUM:
                    correct_correct.add(i)
            else:
                if len(wrong_wrong) < MAX_NUM:
                    wrong_wrong.add(i)
        else:
            if result_one == label_y:
                if len(correct_wrong) < MAX_NUM:
                    correct_wrong.add(i)
            elif result_two == label_y:
                if len(wrong_correct) < MAX_NUM:
                    wrong_correct.add(i)
            else:
                if len(wrong_wrong) < MAX_NUM:
                    wrong_wrong.add(i)
        
        # check if the result sets are full
        if len(correct_correct) >= MAX_NUM and len(correct_wrong) >= MAX_NUM and len(wrong_correct) >= MAX_NUM and len(wrong_wrong) >= MAX_NUM:
            break

    # return the result sets as lists
    return list(correct_correct), list(correct_wrong), list(wrong_correct), list(wrong_wrong)

def calculate_top_eigenvalues_hessian(dataset, subdataset_list, criterion, IN_DIM, OUT_DIM):
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    top_eigenvalues_list = []
    for j in range(len(model_list)):
        # get one model
        model = model_list[j]
        model.eval()

        # get the training dataset for the model
        x, y = get_mnist_for_one_model(subdataset_list[j])

        if FLAG == True:
            model.cuda()
            x = x.cuda()
            y = y.cuda()
        
        # calculate hessian and top eigenvalues
        hessian_comp = hessian(model, criterion, data=(x, y), cuda=FLAG)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
        top_eigenvalues_list.append(top_eigenvalues)
    return top_eigenvalues_list

def calculate_torchvision_model_top_eigenvalues_hessian(subdataset_list, model_name, num_classes, criterion):
    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)
    top_eigenvalues_list = []
    
    # get each model and its training dataset and calculate the top eigenvalues for hessian
    for i in range(len(model_list)):
        # get one model
        model = model_list[i]
        model.eval()

        # prepare the training dataset for top eigenvalues for hessian
        x, y = get_cifar10_for_one_model(subdataset_list[i])

        if FLAG == True:
            model.cuda()
            x = x.cuda()
            y = y.cuda()
        
        # calculate hessian and top eigenvalues
        print("Start Calculating Top Eigenvalues for Hessian for the model: ", model_name, " on the sub-dataset: ", subdataset_list[i], " ...")
        hessian_comp = hessian(model, criterion, data=(x, y), cuda=FLAG)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
        print("The top eigenvalues for the model: ", model_name, " on the sub-dataset: ", subdataset_list[i], " are: ", top_eigenvalues)
        top_eigenvalues_list.append(top_eigenvalues)

    return top_eigenvalues_list

def calculate_vit_model_loss_landscapes_random_projection(dataset_name, subdataset_list, model_name, criterion, x, y, STEPS):
    # prepare the result list
    loss_data_fin_list = []
    model_info_list = []
    max_loss_value_list = []
    min_loss_value_list = []

    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)

    # calculate the first original sub-dataset and get one random projection
    metric = sl_metrics.LossTorchvision(criterion, x, y)
    
    # calculate the loss landscape in 2 dimensions for all models
    for i in range(len(model_list)):
        loss_data_fin, model_info, dir_one, dir_two = loss_landscapes_main.random_plane(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_array = np.array(loss_data_fin)
        loss_data_fin_array = np.where(np.isnan(loss_data_fin_array), ma.array(loss_data_fin_array, mask=np.isnan(loss_data_fin_array)).max(axis=1), loss_data_fin_array)
        loss_data_fin = loss_data_fin_array.tolist()
        loss_data_fin_list.append(loss_data_fin)
        model_info_list.append(model_info)

        # first array corresponds to which row, and the latter corresponds to which column
        max_loss = np.where(loss_data_fin == np.max(loss_data_fin))
        max_loss_value_list.append(max_loss[0][0])
        max_loss_value_list.append(max_loss[1][0])
        min_loss = np.where(loss_data_fin == np.min(loss_data_fin))
        min_loss_value_list.append(min_loss[0][0])
        min_loss_value_list.append(min_loss[1][0])
    
    return loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list

def get_vit_model_list(dataset_name, subdataset_list, model_name):
    # load all the models and store them into a list
    model_list = []
    for subdataset in subdataset_list:
        threshold = 'threshold_{100}'
        if subdataset == 'all':
            threshold = 'threshold_{100}'
        elif subdataset == 'original':
            threshold = 'threshold_{00}'
        else:
            threshold = subdataset
        model_path = '../model/' + dataset_name + '/' + model_name + '/' + model_name + '_model_' + threshold + '.pt'
        this_model = ViT(image_size = 32, patch_size = 4, num_classes = 10, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)
        this_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        this_model.eval()
        model_list.append(this_model)
    return model_list

def get_mnist_for_one_model(subdataset):
    if subdataset == 'original':
        mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
    elif subdataset == 'brightness':
        x_c = load('../data/MNIST_C/brightness/train_images.npy')
        y_c = load('../data/MNIST_C/brightness/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'canny_edges':
        x_c = load('../data/MNIST_C/canny_edges/train_images.npy')
        y_c = load('../data/MNIST_C/canny_edges/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'dotted_line':
        x_c = load('../data/MNIST_C/dotted_line/train_images.npy')
        y_c = load('../data/MNIST_C/dotted_line/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'fog':
        x_c = load('../data/MNIST_C/fog/train_images.npy')
        y_c = load('../data/MNIST_C/fog/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'glass_blur':
        x_c = load('../data/MNIST_C/glass_blur/train_images.npy')
        y_c = load('../data/MNIST_C/glass_blur/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'identity':
        x_c = load('../data/MNIST_C/identity/train_images.npy')
        y_c = load('../data/MNIST_C/identity/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'impulse_noise':
        x_c = load('../data/MNIST_C/impulse_noise/train_images.npy')
        y_c = load('../data/MNIST_C/impulse_noise/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'motion_blur':
        x_c = load('../data/MNIST_C/motion_blur/train_images.npy')
        y_c = load('../data/MNIST_C/motion_blur/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'rotate':
        x_c = load('../data/MNIST_C/rotate/train_images.npy')
        y_c = load('../data/MNIST_C/rotate/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'scale':
        x_c = load('../data/MNIST_C/scale/train_images.npy')
        y_c = load('../data/MNIST_C/scale/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'shear':
        x_c = load('../data/MNIST_C/shear/train_images.npy')
        y_c = load('../data/MNIST_C/shear/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'shot_noise':
        x_c = load('../data/MNIST_C/shot_noise/train_images.npy')
        y_c = load('../data/MNIST_C/shot_noise/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'spatter':
        x_c = load('../data/MNIST_C/spatter/train_images.npy')
        y_c = load('../data/MNIST_C/spatter/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'stripe':
        x_c = load('../data/MNIST_C/stripe/train_images.npy')
        y_c = load('../data/MNIST_C/stripe/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'translate':
        x_c = load('../data/MNIST_C/translate/train_images.npy')
        y_c = load('../data/MNIST_C/translate/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'zigzag':
        x_c = load('../data/MNIST_C/zigzag/train_images.npy')
        y_c = load('../data/MNIST_C/zigzag/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        mnist_train = []
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    elif subdataset == 'all':
        mnist_original = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
        mnist_original_train_loader = torch.utils.data.DataLoader(mnist_original, batch_size=MNIST_BATCH_SIZE, shuffle=False)
        x, y = iter(mnist_original_train_loader).__next__()
        mnist_train = []
        for i in range(len(x)):
            mnist_train.append([x[i], y[i]])
        x_c = load('../data/MNIST_C/brightness/train_images.npy')
        y_c = load('../data/MNIST_C/brightness/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/canny_edges/train_images.npy')
        y_c = load('../data/MNIST_C/canny_edges/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/dotted_line/train_images.npy')
        y_c = load('../data/MNIST_C/dotted_line/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/fog/train_images.npy')
        y_c = load('../data/MNIST_C/fog/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/glass_blur/train_images.npy')
        y_c = load('../data/MNIST_C/glass_blur/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/identity/train_images.npy')
        y_c = load('../data/MNIST_C/identity/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/impulse_noise/train_images.npy')
        y_c = load('../data/MNIST_C/impulse_noise/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/motion_blur/train_images.npy')
        y_c = load('../data/MNIST_C/motion_blur/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/rotate/train_images.npy')
        y_c = load('../data/MNIST_C/rotate/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/scale/train_images.npy')
        y_c = load('../data/MNIST_C/scale/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/shear/train_images.npy')
        y_c = load('../data/MNIST_C/shear/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/shot_noise/train_images.npy')
        y_c = load('../data/MNIST_C/shot_noise/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/spatter/train_images.npy')
        y_c = load('../data/MNIST_C/spatter/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/stripe/train_images.npy')
        y_c = load('../data/MNIST_C/stripe/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/translate/train_images.npy')
        y_c = load('../data/MNIST_C/translate/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
        x_c = load('../data/MNIST_C/zigzag/train_images.npy')
        y_c = load('../data/MNIST_C/zigzag/train_labels.npy')
        x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
        y_c = torch.from_numpy(y_c.reshape(60000)).long()
        for i in range(len(x_c)):
            mnist_train.append([x_c[i], y_c[i]])
    
    # define training data loader
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=MNIST_BATCH_SIZE, shuffle=False)
    
    # set x and y from test_loader
    x, y = iter(train_loader).__next__()

    return x, y

def get_cifar10_for_one_model(subdataset):
    # prepare the training dataset for top eigenvalues for hessian
    cifar10_training = []
    threshold = 'threshold_{100}'
    if subdataset == 'all':
        threshold = 'threshold_{100}'
    elif subdataset == 'original':
        threshold = 'threshold_{00}'
    else:
        threshold = subdataset
    CIFAR10C_PERCENTAGE = int(int(threshold.split('_')[1].split('{')[1].split('}')[0])/10000)
    x_augmented, y_augmented = load_cifar10c(n_examples=int(CIFAR10_SIZE*CIFAR10C_PERCENTAGE), data_dir='../data/')
    for i in range(len(x_augmented)):
        cifar10_training.append([x_augmented[i].float(), y_augmented[i].long()])
    cifar10_original = torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    cifar10_original_loader  = torch.utils.data.DataLoader(cifar10_original, batch_size=1,drop_last=False)
    cifar10_original_iterator = iter(cifar10_original_loader)
    for i in range(int(len(cifar10_original_loader)/100)):
        x_original, y_original = cifar10_original_iterator.__next__()
        cifar10_training.append([x_original, y_original])

    # define training data loader
    train_loader = torch.utils.data.DataLoader(cifar10_training, batch_size=1, shuffle=False)

    # set iterator to test_loader
    test_loader_iter = iter(train_loader)
    
    x_array = []
    y_array = []
    
    # set x and y from test_loader_iter
    for i in range(len(train_loader)):
        this_x, this_y = test_loader_iter.__next__()
        this_x = this_x.reshape(3, 32, 32)
        x_array.append(this_x.numpy())
        y_array.append(this_y.item())
    
    x = torch.tensor(np.array(x_array))
    y = torch.tensor(np.array(y_array))

    return x, y

def calculate_training_loss_landscape(dataset, subdataset_list, criterion, IN_DIM, OUT_DIM, STEPS):
    # prepare the result list
    loss_data_fin_list = []
    max_loss_value_list = []
    min_loss_value_list = []
    model_info_list = []
    
    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    
    # calculate the loss landscape in 2 dimensions for all models
    for i in range(len(model_list)):
        # prepare the training data based on the model
        x, y = get_mnist_for_one_model(subdataset_list[i])

        # prepare the metric
        metric = loss_landscapes.metrics.Loss(criterion, x, y)
        
        # calculate the loss landscape in 2 dimensions for the model
        loss_data_fin_this, model_info_this, dir_one, dir_two = loss_landscapes_main.random_plane(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_this)
        model_info_list.append(model_info_this)

        # first array corresponds to which row, and the latter corresponds to which column
        max_loss_this = np.where(loss_data_fin_this == np.max(loss_data_fin_this))
        max_loss_value_list.append(max_loss_this[0][0])
        max_loss_value_list.append(max_loss_this[1][0])
        min_loss_this = np.where(loss_data_fin_this == np.min(loss_data_fin_this))
        min_loss_value_list.append(min_loss_this[0][0])
        min_loss_value_list.append(min_loss_this[1][0])
        
    return loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list

def calculate_torchvision_model_training_loss_landscapes(subdataset_list, model_name, num_classes, criterion, STEPS):
    # prepare the result list
    loss_data_fin_list = []
    model_info_list = []
    max_loss_value_list = []
    min_loss_value_list = []

    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)

    # calculate the loss landscape in 2 dimensions for the rest of the models
    for i in range(len(model_list)):
        # prepare the training data based on the model
        x, y = get_cifar10_for_one_model(subdataset_list[i])

        # prepare the metric
        metric = sl_metrics.LossTorchvision(criterion, x, y)
        # metric = sl_metrics.Loss(criterion, x, y)

        # calculate the loss landscape in 2 dimensions for all models
        loss_data_fin_this, model_info_this, dir_one, dir_two = loss_landscapes_main.random_plane_rmbn2(model_list[i], metric, 0.1, STEPS, normalization='model', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_this)
        model_info_list.append(model_info_this)

        # first array corresponds to which row, and the latter corresponds to which column
        max_loss_this = np.where(loss_data_fin_this == np.max(loss_data_fin_this))
        max_loss_value_list.append(max_loss_this[0][0])
        max_loss_value_list.append(max_loss_this[1][0])
        min_loss_this = np.where(loss_data_fin_this == np.min(loss_data_fin_this))
        min_loss_value_list.append(min_loss_this[0][0])
        min_loss_value_list.append(min_loss_this[1][0])
        
    return loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list

def calculate_torchvision_model_layer_cka_similarity(subdataset_list, model_name, num_classes, layer_name, linear_reshape_x, linear_reshape_y, kernel_reshape_x, kernel_reshape_y):
    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)

    all_model_layer_list = []
    # get the layer weights of all models
    for model in model_list:
        current_model_layer_list = []
        for name, param in model.named_parameters():
            if 'conv1' in name and 'weight' in name and layer_name in name:
                # print(name)
                # print(param.data.numpy().shape)
                current_model_layer_list.append(param.data)
        all_model_layer_list.append(current_model_layer_list)
    
    print("Finished getting all the layer weights of all models.")

    # initialize the CKA class
    np_cka = CKA()

    # calculate the linear CKA similarity for layers
    linear_cka_similarity_column_all_layer = []
    for k in range(len(all_model_layer_list[0])):
        linear_cka_similarity_column_current_layer = []
        for i in range(len(all_model_layer_list)):
            linear_cka_similarity_row_current_layer = []
            for j in range(len(all_model_layer_list)):
                # print(all_model_layer_list[i][k].numpy().shape)
                # print(all_model_layer_list[j][k].numpy().shape)
                linear_cka_similarity_row_current_layer.append(np_cka.linear_CKA(all_model_layer_list[i][k].numpy().reshape(linear_reshape_x,linear_reshape_y), all_model_layer_list[j][k].numpy().reshape(linear_reshape_x,linear_reshape_y)))
            linear_cka_similarity_column_current_layer.append(linear_cka_similarity_row_current_layer)
        linear_cka_similarity_column_all_layer.append(linear_cka_similarity_column_current_layer)
    
    # calculate the kernel CKA similarity for layers
    kernel_cka_similarity_column_all_layer = []
    for k in range(len(all_model_layer_list[0])):
        kernel_cka_similarity_column_current_layer = []
        for i in range(len(all_model_layer_list)):
            kernel_cka_similarity_row_current_layer = []
            for j in range(len(all_model_layer_list)):
                # print(all_model_layer_list[i][k].numpy().shape)
                # print(all_model_layer_list[j][k].numpy().shape)
                kernel_cka_similarity_row_current_layer.append(np_cka.kernel_CKA(all_model_layer_list[i][k].numpy().reshape(kernel_reshape_x,kernel_reshape_y), all_model_layer_list[j][k].numpy().reshape(kernel_reshape_x,kernel_reshape_y)))
            kernel_cka_similarity_column_current_layer.append(kernel_cka_similarity_row_current_layer)
        kernel_cka_similarity_column_all_layer.append(kernel_cka_similarity_column_current_layer)
    
    # generate the distance matrix for linear CKA similarity for layers
    linear_cka_similarity_matrix_all_layer = []
    for i in range(len(linear_cka_similarity_column_all_layer)):
        linear_cka_similarity_matrix_all_layer.append(np.array(linear_cka_similarity_column_all_layer[i]))
    
    # generate the distance matrix for kernel CKA similarity for layers
    kernel_cka_similarity_matrix_all_layer = []
    for i in range(len(kernel_cka_similarity_column_all_layer)):
        kernel_cka_similarity_matrix_all_layer.append(np.array(kernel_cka_similarity_column_all_layer[i]))

    return linear_cka_similarity_matrix_all_layer, kernel_cka_similarity_matrix_all_layer

def calculate_training_3d_loss_landscapes_random_projection(dataset, subdataset_list, criterion, IN_DIM, OUT_DIM, STEPS):
    # prepare the result list
    loss_data_fin_list = []

    # prepare all the models
    model_list = get_model_list(dataset, subdataset_list, IN_DIM, OUT_DIM)
    
    # calculate the loss landscape in 3 dimensions for all models
    for i in range(len(model_list)):
        # prepare the training data based on the model
        x, y = get_mnist_for_one_model(subdataset_list[i])

        # prepare the metric
        metric = loss_landscapes.metrics.Loss(criterion, x, y)

        # compute loss landscape 3D data for the rest of the models
        loss_data_fin_3d, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_3d)

    return loss_data_fin_list

def calculate_vit_similarity_global_structure(dataset_name, subdataset_list, model_name):
    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)

    # join the tensors of all subdatasets
    modelTensor_list = []
    for model in model_list:
        modelTensor_list.append(torch.cat((model.to_patch_embedding[2].weight.reshape(-1),model.mlp_head[1].weight.reshape(-1))))
    
    # calculate the euclidean distance between models
    euclidean_distance_column = []
    for i in range(len(model_list)):
        euclidean_distance_row = []
        for j in range(len(model_list)):
            euclidean_distance_row.append((modelTensor_list[i] - modelTensor_list[j]).pow(2).sum().sqrt().item())
        euclidean_distance_column.append(euclidean_distance_row)

    # generate the distance matrix
    print(type(euclidean_distance_column))
    euclidean_distance_matrix = np.array(euclidean_distance_column)

    # calculate the MDS of the euclidean distance matrix for models similarity
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(euclidean_distance_matrix)

    # calculate the global structure of the models based on the MDS
    condensed_distance_matrix = pdist(X_transformed)
    Z = hierarchy.linkage(condensed_distance_matrix, 'single')
    T = to_tree(Z, rd=False)
    figure  = to_newick(T, ascii_lowercase)
    
    return X_transformed, figure

def calculate_vit_cka_similarity_global_structure(dataset_name, subdataset_list, model_name, reshape_x, reshape_y):
    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)

    # join the tensors of all subdatasets
    modelTensor_list = []
    for model in model_list:
        modelTensor_list.append(torch.cat((model.to_patch_embedding[2].weight.reshape(-1),model.mlp_head[1].weight.reshape(-1))))
    
    # initialize the numpy CKA class
    np_cka = CKA()
    
    # calculate the linear CKA similarity between models
    linear_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        linear_cka_similarity_row = []
        for j in range(len(model_list)):
            linear_cka_similarity_row.append(np_cka.linear_CKA(modelTensor_list[i].detach().numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].detach().numpy().reshape((reshape_x, reshape_y))))
        linear_cka_similarity_column.append(linear_cka_similarity_row)
    
    kernel_cka_similarity_column = []
    pbar = tqdm(model_list)
    for k in pbar:
        i = model_list.index(k)
        kernel_cka_similarity_row = []
        for j in range(len(model_list)):
            kernel_cka_similarity_row.append(np_cka.kernel_CKA(modelTensor_list[i].detach().numpy().reshape((reshape_x, reshape_y)), modelTensor_list[j].detach().numpy().reshape((reshape_x, reshape_y))))
        kernel_cka_similarity_column.append(kernel_cka_similarity_row)
    
    # generate the distance matrix for linear CKA
    linear_cka_matrix = 1 - np.array(linear_cka_similarity_column)

    # generate the distance matrix for RBF kernel CKA
    kernel_cka_matrix = 1 - np.array(kernel_cka_similarity_column)
    
    # calculate the MDS of the linear CKA distance matrix for models similarity
    linear_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    linear_cka_embedding = linear_cka_mds.fit_transform(linear_cka_matrix)

    # calculate the global structure of the models based on the linear CKA MDS
    linear_cka_condensed_distance_matrix = pdist(linear_cka_embedding)
    linear_cka_Z = hierarchy.linkage(linear_cka_condensed_distance_matrix, 'single')
    linear_cka_T = to_tree(linear_cka_Z, rd=False)
    linear_cka_figure  = to_newick(linear_cka_T, ascii_lowercase)

    # calculate the MDS of the RBF kernel CKA distance matrix for models similarity
    kernel_cka_mds = MDS(n_components=2, dissimilarity='precomputed')
    kernel_cka_embedding = kernel_cka_mds.fit_transform(kernel_cka_matrix)

    # calculate the global structure of the models based on the RBF kernel CKA MDS
    kernel_cka_condensed_distance_matrix = pdist(kernel_cka_embedding)
    kernel_cka_Z = hierarchy.linkage(kernel_cka_condensed_distance_matrix, 'single')
    kernel_cka_T = to_tree(kernel_cka_Z, rd=False)
    kernel_cka_figure  = to_newick(kernel_cka_T, ascii_lowercase)

    return linear_cka_embedding, kernel_cka_embedding, linear_cka_figure, kernel_cka_figure

def calculate_vit_layer_torch_cka_similarity(dataset_name, subdataset_list, model_name):
    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)

    # prepare the test dataloader
    cifar10_testing = []
    x_augmented, y_augmented = load_cifar10c(n_examples=int(CIFAR10_SIZE*CIFAR10C_PERCENTAGE_LAYER_CKA), data_dir='../data/')
    for i in range(len(x_augmented)):
        cifar10_testing.append([x_augmented[i].float().reshape(3, 32, 32), y_augmented[i].long()])
    cifar10_original = torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    cifar10_original_loader  = torch.utils.data.DataLoader(cifar10_original, batch_size=1,drop_last=False)
    cifar10_original_iterator = iter(cifar10_original_loader)
    for i in range(int(len(cifar10_original_loader)*CIFAR10C_PERCENTAGE_LAYER_CKA)):
        x_original, y_original = cifar10_original_iterator.__next__()
        x_original = torch.tensor(np.array(x_original.reshape(3, 32, 32)))
        cifar10_testing.append([x_original, y_original])
    cifar10_original_test = torchvision.datasets.CIFAR10('../data/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    cifar10_original_test_loader  = torch.utils.data.DataLoader(cifar10_original_test, batch_size=1,drop_last=False)
    cifar10_original_test_iterator = iter(cifar10_original_test_loader)
    for i in range(int(len(cifar10_original_test_loader)*CIFAR10C_PERCENTAGE_LAYER_CKA)):
        x_original, y_original = cifar10_original_test_iterator.__next__()
        x_original = torch.tensor(np.array(x_original.reshape(3, 32, 32)))
        cifar10_testing.append([x_original, y_original])

    # define testing data loader
    dataloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=CIFAR10_BATCH_SIZE_LAYER_CKA, shuffle=False)

    # calculate the CKA distance for models using torch CKA
    cka_result_all = []
    for i in range(len(model_list)):
        cka_result_row = []
        for j in range(len(model_list)):
            cka = torch_cka.CKA(model_list[i], model_list[j], device=DEVICE)
            cka.compare(dataloader)
            results = cka.export()
            cka_result_row.append(results)
        cka_result_all.append(cka_result_row)

    return cka_result_all

def calculate_torchvision_layer_torch_cka_similarity(subdataset_list, model_name, num_classes):
    # prepare all the models
    model_list = get_torchvision_model_list(subdataset_list, model_name, num_classes)

    # prepare the test dataloader
    cifar10_testing = []
    x_augmented, y_augmented = load_cifar10c(n_examples=int(CIFAR10_SIZE*CIFAR10C_PERCENTAGE_LAYER_CKA), data_dir='../data/')
    for i in range(len(x_augmented)):
        cifar10_testing.append([x_augmented[i].float().reshape(3, 32, 32), y_augmented[i].long()])
    cifar10_original = torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    cifar10_original_loader  = torch.utils.data.DataLoader(cifar10_original, batch_size=1,drop_last=False)
    cifar10_original_iterator = iter(cifar10_original_loader)
    for i in range(int(len(cifar10_original_loader)*CIFAR10C_PERCENTAGE_LAYER_CKA)):
        x_original, y_original = cifar10_original_iterator.__next__()
        x_original = torch.tensor(np.array(x_original.reshape(3, 32, 32)))
        cifar10_testing.append([x_original, y_original])
    cifar10_original_test = torchvision.datasets.CIFAR10('../data/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    cifar10_original_test_loader  = torch.utils.data.DataLoader(cifar10_original_test, batch_size=1,drop_last=False)
    cifar10_original_test_iterator = iter(cifar10_original_test_loader)
    for i in range(int(len(cifar10_original_test_loader)*CIFAR10C_PERCENTAGE_LAYER_CKA)):
        x_original, y_original = cifar10_original_test_iterator.__next__()
        x_original = torch.tensor(np.array(x_original.reshape(3, 32, 32)))
        cifar10_testing.append([x_original, y_original])

    # define testing data loader
    dataloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=CIFAR10_BATCH_SIZE_LAYER_CKA, shuffle=False)

    # # define layer name list for torchvision models
    # layer_names_list = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'fc']

    # calculate the CKA distance for models using torch CKA
    cka_result_all = []
    for i in range(len(model_list)):
        cka_result_row = []
        for j in range(len(model_list)):
            # cka = torch_CKA(model_list[i], model_list[j], model1_layers=layer_names_list, model2_layers=layer_names_list, device=DEVICE)
            cka = torch_cka.CKA(model_list[i], model_list[j], device=DEVICE)
            cka.compare(dataloader)
            results = cka.export()
            cka_result_row.append(results)
        cka_result_all.append(cka_result_row)

    return cka_result_all

def calculate_vit_top_eigenvalues_hessian(dataset_name, subdataset_list, model_name, criterion):
    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)
    top_eigenvalues_list = []
    
    # get each model and its training dataset and calculate the top eigenvalues for hessian
    for i in range(len(model_list)):
        # get one model
        model = model_list[i]
        model.eval()

        # prepare the training dataset for top eigenvalues for hessian
        x, y = get_cifar10_for_one_model(subdataset_list[i])

        if FLAG == True:
            model.cuda()
            x = x.cuda()
            y = y.cuda()

        # calculate hessian and top eigenvalues
        hessian_comp = hessian(model, criterion, data=(x, y), cuda=FLAG)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
        top_eigenvalues_list.append(top_eigenvalues)
    return top_eigenvalues_list

def calculate_vit_model_information(dataset_name, subdataset_list, model_name, i, x, y):
    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)
    model = model_list[i]
    model.eval()

    # compute the accuracy, recall, precision, f1 score and confusion matrix of the perb model
    preds = model(x)
    # get the accuracy
    accuracy = Accuracy(task="multiclass", average='macro', num_classes=10)
    model_accuracy = accuracy(preds, y)
    # get the recall
    recall = Recall(task="multiclass", average='macro', num_classes=10)
    model_recall = recall(preds, y)
    # get the precision
    precision = Precision(task="multiclass", average='macro', num_classes=10)
    model_precision = precision(preds, y)
    # get the f1 score
    f1 = F1Score(task="multiclass", num_classes=10)
    model_f1 = f1(preds, y)
    # get the confusion matrix
    confusionMatrix = ConfusionMatrix(task="multiclass", num_classes=10)
    model_confusionMatrix = confusionMatrix(preds, y)

    return model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix

def calculate_vit_training_loss_landscapes(dataset_name, subdataset_list, model_name, criterion, STEPS):
    # prepare the result list
    loss_data_fin_list = []
    model_info_list = []
    max_loss_value_list = []
    min_loss_value_list = []

    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)

    # calculate the loss landscape in 2 dimensions for the rest of the models
    for i in range(len(model_list)):
        # prepare the training data based on the model
        x, y = get_cifar10_for_one_model(subdataset_list[i])

        # prepare the metric
        metric = sl_metrics.LossTorchvision(criterion, x, y)

        # calculate the loss landscape in 2 dimensions for all models
        loss_data_fin_this, model_info_this, dir_one, dir_two = loss_landscapes_main.random_plane(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_this)
        model_info_list.append(model_info_this)

        # first array corresponds to which row, and the latter corresponds to which column
        max_loss_this = np.where(loss_data_fin_this == np.max(loss_data_fin_this))
        max_loss_value_list.append(max_loss_this[0][0])
        max_loss_value_list.append(max_loss_this[1][0])
        min_loss_this = np.where(loss_data_fin_this == np.min(loss_data_fin_this))
        min_loss_value_list.append(min_loss_this[0][0])
        min_loss_value_list.append(min_loss_this[1][0])
        
    return loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list

def calculate_vit_3d_loss_landscapes_random_projection(dataset_name, subdataset_list, model_name, criterion, x, y, STEPS):
    # prepare the result list
    loss_data_fin_list = []

    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)

    # calculate the first original sub-dataset and get one random projection
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    # calculate the loss landscape in 3 dimensions for all models
    for i in range(len(model_list)):
        # compute loss landscape 3D data for the rest of the models
        loss_data_fin_3d, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss_data_fin_list.append(loss_data_fin_3d)
    
    return loss_data_fin_list

def calculate_vit_training_3d_loss_landscapes_random_projection(dataset_name, subdataset_list, model_name, criterion, STEPS):
    # prepare the result list
    loss_data_fin_list = []

    # prepare all the models
    model_list = get_vit_model_list(dataset_name, subdataset_list, model_name)
    
    # calculate the loss landscape in 3 dimensions for all models
    for i in range(len(model_list)):
        # prepare the training data based on the model
        x, y = get_cifar10_for_one_model(subdataset_list[i])

        # prepare the metric
        metric = loss_landscapes.metrics.Loss(criterion, x, y)

        for i in range(len(model_list)):
            # compute loss landscape 3D data for the rest of the models
            loss_data_fin_3d, dir_one_space, dir_two_space, dir_three_space = loss_landscapes.random_space(model_list[i], metric, 10, STEPS, normalization='filter', deepcopy_model=True)
            loss_data_fin_list.append(loss_data_fin_3d)

    return loss_data_fin_list

def calculate_vit_prediction_distribution(model_one, model_two, distribution_test_loader, MAX_NUM):
    # prepare the result set
    correct_correct = set()
    correct_wrong = set()
    wrong_correct = set()
    wrong_wrong = set()

    # predict each testing image one by one
    model_one.eval()
    model_two.eval()

    # create the iterator for the testing dataset
    distribution_test_loader_iter = iter(distribution_test_loader)

    print("Length of the testing dataset: ", len(distribution_test_loader))

    # calculate the prediction distribution result for each testing image
    for i in range(len(distribution_test_loader)):
    # for i in range(10000):
        # get one testing image
        try:
            distribution_x, distribution_y = distribution_test_loader_iter.__next__()
        except StopIteration:
            distribution_test_loader_iter = iter(distribution_test_loader)
            distribution_x, distribution_y = next(distribution_test_loader_iter)
        # prediction by the first model
        distribution_x = distribution_x.reshape(1, 3, 32, 32)
        prediction_one = model_one(distribution_x)
        result_one = torch.argmax(prediction_one, dim=1).item()
        # prediction by the second model
        prediction_two = model_two(distribution_x)
        result_two = torch.argmax(prediction_two, dim=1).item()
        # get the correct label
        label_y = distribution_y.item()
        # print("The correct label is: ", label_y)
        # print("The prediction result for the first model is: ", result_one)
        # print("The prediction result for the second model is: ", result_two)

        # calculate the prediction distribution result for one testing image
        if result_one == result_two:
            if result_one == label_y:
                if len(correct_correct) < MAX_NUM:
                    correct_correct.add(i)
            else:
                if len(wrong_wrong) < MAX_NUM:
                    wrong_wrong.add(i)
        else:
            if result_one == label_y:
                if len(correct_wrong) < MAX_NUM:
                    correct_wrong.add(i)
            elif result_two == label_y:
                if len(wrong_correct) < MAX_NUM:
                    wrong_correct.add(i)
            else:
                if len(wrong_wrong) < MAX_NUM:
                    wrong_wrong.add(i)
        
        # check if the result sets are full
        if len(correct_correct) >= MAX_NUM and len(correct_wrong) >= MAX_NUM and len(wrong_correct) >= MAX_NUM and len(wrong_wrong) >= MAX_NUM:
            break

    # return the result sets as lists
    return list(correct_correct), list(correct_wrong), list(wrong_correct), list(wrong_wrong)

class MinibatchCKA(nn.Module):
    expansion = 1

    def __init__(self, num_layers, device, num_layers2=None, across_models=False):
        super(MinibatchCKA, self).__init__()

        if num_layers2 is None:
            num_layers2 = num_layers
        
        self.hsic_accumulator = torch.zeros(num_layers, num_layers)
        self.hsic_accumulator = self.hsic_accumulator.to(device)
        self.across_models = across_models
        if across_models:
            self.hsic_accumulator_model1 = torch.zeros(num_layers,).to(device)
            self.hsic_accumulator_model2 = torch.zeros(num_layers2,).to(device)
            #print(self.hsic_accumulator_model1.shape, self.hsic_accumulator_model2.shape )
            
        print(self.hsic_accumulator.shape)

    def _generate_gram_matrix(self, x):
        """Generate Gram matrix and preprocess to compute unbiased HSIC.
        https://inst.eecs.berkeley.edu/~ee127/sp21/livebook/def_Gram_matrix.html
        This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
        L. (2014). Partial distance correlation with methods for dissimilarities.
        The Annals of Statistics, 42(6), 2382-2412.
        """
        #print("_generate_gram_matrix")
        x = torch.reshape(x, (x.shape[0], -1))
        gram = torch.matmul(x, x.T)
        n = gram.shape[0]
        gram.fill_diagonal_(0)
        means = torch.sum(gram, axis=0) / (n - 2)
        means = means - ( torch.sum(means) /  (2 * (n-1))   )

        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)
        gram = torch.reshape(gram, (-1,))

        return gram

    def update_state(self, activations):
        """
        Accumulate minibatch HSIC values.
        Args:
            activations: a list of activations for all layers
        """
        #for x in activations:
        #    grams = self._generate_gram_matrix(x)
        #    print(grams.shape)
        # for idx, x in enumerate(activations):
        #     if idx == 5:
        #         print("activations", x.sum())
        layer_grams = [self._generate_gram_matrix(x) for x in activations]
        layer_grams = torch.stack(layer_grams, dim=0)
        hsic = torch.matmul(layer_grams, layer_grams.T)
    
        self.hsic_accumulator += hsic
        #print(layer_grams[5].shape, layer_grams[5])
        #import sys
        #sys.exit(0)
        #print(hsic.detach().cpu().numpy()[5])
   
    def result(self):
        mean_hsic = self.hsic_accumulator
        #print(mean_hsic.detach().cpu().numpy()[5])
        if self.across_models:
            raise NotImplementedError
        else:
            normalization = torch.sqrt(torch.diag(mean_hsic, 0))
            mean_hsic /= normalization[:, None]
            mean_hsic /= normalization[None, :]
        #print(mean_hsic.detach().cpu().numpy()[19])
        return mean_hsic
