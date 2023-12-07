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

# from pyevtk.hl import gridToVTK
from pyevtk.hl import imageToVTK

from operation import functions as operation

# training hyperparameters
batch_size_train = 64
batch_size_test = 128
num_classes = 10
CIFAR10_SIZE = 60000
CIFAR10C_PERCENTAGE = 0.01
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
parser.add_argument('--model', default='VIT', help='trained model for testing')
parser.add_argument('--reshapex', default=464, help='reshape x for cka distance calculation')
parser.add_argument('--reshapey', default=128, help='reshape y for cka distance calculation')
parser.add_argument('--losssteps', default=4, help='steps for loss landscape calculation')
parser.add_argument('--device', default="local", help='device to do the back-end calculation')
args = parser.parse_args()

# Loss Landscape parameters
LOSS_STEPS = int(args.losssteps)
# Model Name
model_name = args.model
# set the subdataset list
subdataset_list = ['original', 'threshold_{20}', 'threshold_{40}', 'threshold_{60}', 'threshold_{80}', 'all']

# set the client and database name
client = "mongodb://localhost:27017/"
database_name = "mydatabase"
if args.device == "nersc":
    client = "mongodb07.nersc.gov"
    database_name = "losslensdb"

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
collection_name_layer_cka_similarity = "layer_cka_similarity"
collection_name_layer_torch_cka_similarity = "layer_torch_cka_similarity"

# define loss function
criterion = torch.nn.CrossEntropyLoss()

# calculate the euclidean distance and global structure between models
print("Calculating the euclidean distance between models...")
X_transformed, global_structure = operation.calculate_vit_similarity_global_structure(dataset_name,subdataset_list,model_name)
print("Finished calculating the euclidean distance between models.")

# print the euclidean distance matrix and the global structure
print("Euclidean distance between models: ", X_transformed)
print("Euclidean distance matrix shape: ", X_transformed.shape)

# save the model euclidean distance similarity results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'modelMDS': X_transformed.tolist()}
record_id = operation.store_into_database(client, database_name, collection_name_model_euclidean_distance_similarity,record)
print("One record of model_euclidean_distance_similarity has been successfully inserted with ID " + str(record_id))

# calculate the cka distance between models
print("Calculating the CKA distance between models...")
reshape_x = int(args.reshapex)
reshape_y = int(args.reshapey)
linear_cka_embedding, kernel_cka_embedding, linear_cka_figure, kernel_cka_figure = operation.calculate_vit_cka_similarity_global_structure(dataset_name, subdataset_list, model_name, reshape_x, reshape_y)
print("Finished calculating the CKA distance between models.")

# print the CKA distances matrix and the MDS results
print("Linear CKA similarity distance between models: ", linear_cka_embedding)
print("Linear CKA similarity matrix shape: ", linear_cka_embedding.shape)
print("RBF kernel CKA similarity distance between models: ", kernel_cka_embedding)
print("RBF kernel CKA similarity matrix shape: ", kernel_cka_embedding.shape)

# save the model CKA similarity results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'linearCKA_MDS': linear_cka_embedding.tolist(),'kernelCKA_MDS': kernel_cka_embedding.tolist()}
record_id = operation.store_into_database(client, database_name, collection_name_cka_model_similarity, record)
print("One record of cka_model_similarity has been successfully inserted with ID " + str(record_id))
print("CKA model similarity information have been saved into MongoDB.")

# print the global structure
print("Global structure of all the models based on Euclidean distance: ", global_structure)
print("Global structure of all the models based on Linear CKA similarity: ", linear_cka_figure)
print("Global structure of all the models based on RBF kernel CKA similarity: ", kernel_cka_figure)

# save the model global structure results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'global_structure': global_structure, 'linearCKA_global_structure': linear_cka_figure, 'kernelCKA_global_structure': kernel_cka_figure}
record_id = operation.store_into_database(client, database_name, collection_name_global_structure, record)
print("One record of global_structure has been successfully inserted with ID " + str(record_id))
print("Model global structure information have been saved into MongoDB.")

# calculate the torch CKA similarity between layers
print("Calculating the CKA similarity between model layers...")
cka_result_all = operation.calculate_vit_layer_torch_cka_similarity(dataset_name, subdataset_list, model_name)
print("Finished calculating the CKA similarity between model layers.")

# print the torch CKA similarity between models
for i in range(len(cka_result_all)):
    cka_result = cka_result_all[i]
    for j in range(len(cka_result)):
        print("Torch CKA similarity between model " + str(i) + " and model " + str(j) + " is: ", cka_result[j])

# save the torch CKA similarity into database
for i in range(len(cka_result_all)):
    cka_result = cka_result_all[i]
    for j in range(len(cka_result)):
        record = {'dataset': dataset_name, 'model_type': model_name, 'model1': str(i), 'model2': str(j), 'torchCKA_similarity': str(cka_result[j])}
        record_id = operation.store_into_database(client, database_name, collection_name_layer_torch_cka_similarity, record)
        print("One record of layer_torch_cka_similarity has been successfully inserted with ID " + str(record_id))
print("Layer torch CKA similarity information have been saved into MongoDB.")

# calculate the top eigenvalues for hessian
print("Calculating the top eigenvalues for hessian...")
top_eigenvalues_list = operation.calculate_vit_top_eigenvalues_hessian(dataset_name, subdataset_list, model_name, criterion)
print("Finished calculating the top eigenvalues for hessian.")

# calculate the training loss landscape
print("Calculating the training loss landscape...")
loss_data_fin_list_train, model_info_list_train, max_loss_value_list_train, min_loss_value_list_train = operation.calculate_vit_training_loss_landscapes(dataset_name, subdataset_list, model_name, criterion, LOSS_STEPS)
loss_data_fin_array_train = np.array(loss_data_fin_list_train)
loss_data_fin_array_train = np.where(np.isnan(loss_data_fin_array_train), ma.array(loss_data_fin_array_train, mask=np.isnan(loss_data_fin_array_train)).mean(axis=0), loss_data_fin_array_train)
loss_data_fin_array_train = np.where(np.isposinf(loss_data_fin_array_train), ma.array(loss_data_fin_array_train, mask=np.isposinf(loss_data_fin_array_train)).max(axis=0), loss_data_fin_array_train)
loss_data_fin_array_train = np.where(np.isneginf(loss_data_fin_array_train), ma.array(loss_data_fin_array_train, mask=np.isneginf(loss_data_fin_array_train)).min(axis=0), loss_data_fin_array_train)
loss_data_fin_list_train = loss_data_fin_array_train.tolist()
print("Finished calculating the training loss landscape.")

# save the single model results to the database
for i in range(len(loss_data_fin_list_train)):
    record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS, "loss_data_fin": loss_data_fin_list_train[i]}
    record_id = operation.store_into_database(client, database_name, collection_name_training_loss_landscape, record)
    print("One record of training_loss_landscape has been successfully inserted with ID " + str(record_id))

# prepare testing data used for evaluating loss landscape
print("Loading CIFAR10 and CIFAR10-C dataset for evaluating loss...")
cifar10_testing = []
x_augmented, y_augmented = load_cifar10c(n_examples=int(CIFAR10_SIZE*CIFAR10C_PERCENTAGE), data_dir='../data/')
for i in range(len(x_augmented)):
    cifar10_testing.append([x_augmented[i].float(), y_augmented[i].long()])
cifar10_original = torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_loader  = torch.utils.data.DataLoader(cifar10_original, batch_size=1,drop_last=False)
cifar10_original_iterator = iter(cifar10_original_loader)
for i in range(int(len(cifar10_original_loader)*CIFAR10C_PERCENTAGE)):
    x_original, y_original = cifar10_original_iterator.__next__()
    cifar10_testing.append([x_original, y_original])
cifar10_original_test = torchvision.datasets.CIFAR10('../data/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_test_loader  = torch.utils.data.DataLoader(cifar10_original_test, batch_size=1,drop_last=False)
cifar10_original_test_iterator = iter(cifar10_original_test_loader)
for i in range(int(len(cifar10_original_test_loader)*CIFAR10C_PERCENTAGE)):
    x_original, y_original = cifar10_original_test_iterator.__next__()
    cifar10_testing.append([x_original, y_original])

# define testing data loader
test_loader = torch.utils.data.DataLoader(cifar10_testing, batch_size=1, shuffle=False)

# set iterator to test_loader
test_loader_iter = iter(test_loader)

x_array = []
y_array = []

# set x and y from test_loader_iter
for i in range(len(test_loader)):
    this_x, this_y = test_loader_iter.__next__()
    this_x = this_x.reshape(3, 32, 32)
    x_array.append(this_x.numpy())
    y_array.append(this_y.item())

x = torch.tensor(np.array(x_array))
y = torch.tensor(np.array(y_array))

# calculate the loss landscapes for random projection
print("Calculating the loss landscapes for random projection...")
loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list = operation.calculate_vit_model_loss_landscapes_random_projection(dataset_name, subdataset_list, model_name, criterion, x, y, LOSS_STEPS)
loss_data_fin_array = np.array(loss_data_fin_list)
loss_data_fin_array[np.isinf(loss_data_fin_array)] = np.nan
loss_data_fin_array = np.where(np.isnan(loss_data_fin_array),ma.array(loss_data_fin_array, mask=np.isnan(loss_data_fin_array)).max(axis=1),loss_data_fin_array)
loss_data_fin_list = loss_data_fin_array.tolist()
print("Finish calculating the loss landscapes for random projection.")

# calculate and save the model analysis information
print("Calculating the model analysis information...")
for i in range(len(subdataset_list)):
    model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix = operation.calculate_vit_model_information(dataset_name, subdataset_list, model_name, i, x, y)
    record = {'dataset': dataset_name, 'model_type': model_name, 'model': subdataset_list[i],'accuracy': model_accuracy.tolist(), 'recall': model_recall.tolist(),'precision': model_precision.tolist(), 'f1': model_f1.tolist(),'confusionMatrix': model_confusionMatrix.tolist(), 'top_eigenvalues': top_eigenvalues_list[i]}
    record_id = operation.store_into_database(client, database_name, collection_name_model_analysis_information, record)
    print("One record of model_analysis_information has been successfully inserted with ID " + str(record_id))
print("Finished calculating the model analysis information.")

# prepare the calculation parameters
DETAILED_START = int(1)
DETAILED_END = int(LOSS_STEPS)
DETAILED_STEP = int(1)
if args.device == "nersc":
    DETAILED_START = int(5)
    DETAILED_END = int(LOSS_STEPS / 2)
    DETAILED_STEP = int(5)

# prepare the model list for detailed similarity calculation
print("Preparing the model list for detailed similarity calculation...")
selected_model_list_all_steps = []
for i in range(DETAILED_START, DETAILED_END, DETAILED_STEP):
    selected_model_list_all_steps.append(operation.get_selected_center_model_list_from_loss_landscapes(model_info_list, i, i, LOSS_STEPS))
print("Finished preparing the model list for detailed similarity calculation.")

# calculate the detailed similarity between loss landscapes models and save the results to the database
print("Calculating the detailed similarity between loss landscapes models...")
for k in range(DETAILED_START, DETAILED_END, DETAILED_STEP):
    index = int(k/DETAILED_STEP) - 1
    selected_model_list_all = selected_model_list_all_steps[index]
    for i in range(len(selected_model_list_all)):
        for j in range(len(selected_model_list_all)):
            this_X_transformed = operation.calculate_detailed_similarity_from_loss_landscapes_models(selected_model_list_all, i, j)
            record = {"dataset": dataset_name, 'model_type': model_name, "model1": subdataset_list[i], "model2": subdataset_list[j], "modelMDS": this_X_transformed.tolist(), "threshold": k}
            record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_detailed_similarity, record)
            print("One record of loss_landscapes_detailed_similarity has been successfully inserted with ID " + str(record_id))
print("Finished calculating the detailed similarity between loss landscapes models.")

# save the single model results to the database
for i in tqdm(range(len(loss_data_fin_list)), desc="Working on loss landscapes for random projection"):
    record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS,"loss_data_fin": loss_data_fin_list[i]}
    record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_contour, record)
    print("One record of loss_landscapes_contour has been successfully inserted with ID " + str(record_id))

# save the global result to the database
record = {"dataset": dataset_name, 'model_type': model_name, "steps": LOSS_STEPS, "loss_data_fin": loss_data_fin_array.tolist()}
record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_global, record)
print("One record of loss_landscapes_global has been successfully inserted with ID " + str(record_id))

# generate 3D loss landscapes for random projection
print("Generating 3D loss landscapes for random projection...")
loss_data_fin_list_3d = operation.calculate_vit_3d_loss_landscapes_random_projection(dataset_name, subdataset_list, model_name, criterion, x, y, LOSS_STEPS)
loss_data_fin_array_3d = np.array(loss_data_fin_list_3d)
loss_data_fin_array_3d = np.where(np.isnan(loss_data_fin_array_3d),ma.array(loss_data_fin_array_3d, mask=np.isnan(loss_data_fin_array_3d)).min(axis=0),loss_data_fin_array_3d)
loss_data_fin_list_3d = loss_data_fin_array_3d.tolist()
print("Finish generating 3D loss landscapes for random projection.")

# save the results to the database
for i in tqdm(range(len(loss_data_fin_list_3d)), desc="Working on 3D loss landscapes for random projection"):
    record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS, "loss_data_fin_3d": loss_data_fin_list_3d[i]}
    record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_contour_3d, record)
    print("One record of loss_landscapes_contour_3d has been successfully inserted with ID " + str(record_id))

# prepare the data to store in .vti files for ttk input
nx, ny, nz = LOSS_STEPS - 1, LOSS_STEPS - 1, LOSS_STEPS - 1
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Variables
pressure = np.zeros(ncells).reshape((nx, ny, nz), order='C')

# store the loss landscape results into binary files used for ttk
for i in range(len(loss_data_fin_list_3d)):
    FILE_PATH = '../ttk/input_binary_for_ttk/CIFAR10_' + model_name + '_' + subdataset_list[i] + '_loss_mnist_3d_contour'
    imageToVTK(FILE_PATH, cellData={"Cell": pressure}, pointData={"Loss": np.array(loss_data_fin_list_3d[i])})

# prepare the testing dataset for prediction distribution computation
print("Loading CIFAR10 and CIFAR10-C dataset for prediction distribution...")
cifar10_prediction_distribution = []
x_augmented, y_augmented = load_cifar10c(n_examples=int(CIFAR10_SIZE * INPUT_DATA_PERCENTAGE), data_dir='../data/')
for i in range(len(x_augmented)):
    cifar10_prediction_distribution.append([x_augmented[i].float(), y_augmented[i].long()])
cifar10_original = torchvision.datasets.CIFAR10('../data/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_loader = torch.utils.data.DataLoader(cifar10_original, batch_size=1, drop_last=False)
cifar10_original_iterator = iter(cifar10_original_loader)
for i in range(int(len(cifar10_original_loader) * INPUT_DATA_PERCENTAGE)):
    x_original, y_original = cifar10_original_iterator.__next__()
    cifar10_prediction_distribution.append([x_original, y_original])
cifar10_original_test = torchvision.datasets.CIFAR10('../data/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
cifar10_original_test_loader = torch.utils.data.DataLoader(cifar10_original_test, batch_size=1, drop_last=False)
cifar10_original_test_iterator = iter(cifar10_original_test_loader)
for i in range(int(len(cifar10_original_test_loader) * INPUT_DATA_PERCENTAGE)):
    x_original, y_original = cifar10_original_test_iterator.__next__()
    cifar10_prediction_distribution.append([x_original, y_original])  
# define testing data loader for prediction distribution calculation
distribution_test_loader = torch.utils.data.DataLoader(cifar10_prediction_distribution, batch_size=1, shuffle=False)
print("Finish loading the testing data for prediction distribution calculation.")

# get the models
print("Getting the models for prediction distribution computation...")
subdataset_list = ['original', 'threshold_{20}', 'threshold_{40}', 'threshold_{60}', 'threshold_{80}', 'all']
model_list = model_list = operation.get_vit_model_list(dataset_name, subdataset_list, model_name)
print("Finish getting the models for prediction distribution computation.")

# calculate the prediction distribution of each pair of models
print("Calculating the prediction distribution of all models...")
for i in range(len(model_list)):
    for j in range(len(model_list)):
        # select two models
        model_one = model_list[i]
        model_two = model_list[j]
        # calculate the prediction distribution
        correct_correct, correct_wrong, wrong_correct, wrong_wrong = operation.calculate_vit_prediction_distribution(model_one, model_two, distribution_test_loader, MAX_NUM)
        print("Finish calculating the prediction distribution of one pair of models.")
        print("Correct-Correct: " + str(correct_correct) + ", Correct-Wrong: " + str(correct_wrong) + ", Wrong-Correct: " + str(wrong_correct) + ", Wrong-Wrong: " + str(wrong_wrong))
        # store the results into the database
        record = {"dataset": dataset_name, 'model_type': model_name, "model1": subdataset_list[i], "model2": subdataset_list[j], "correct_correct": correct_correct, "correct_wrong": correct_wrong, "wrong_correct": wrong_correct, "wrong_wrong": wrong_wrong}
        record_id = operation.store_into_database(client, database_name, collection_name_prediction_distribution, record)
        print("One record of prediction_distribution has been successfully inserted with ID " + str(record_id))
print("Finish calculating the prediction distribution of all models.")
