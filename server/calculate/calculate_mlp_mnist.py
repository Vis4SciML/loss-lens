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
parser.add_argument('--losssteps', default=40, help='steps for loss landscape calculation')
parser.add_argument('--device', default="local", help='device to do the back-end calculation')
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
collection_name_layer_torch_cka_similarity = "layer_torch_cka_similarity"

# define the loss function
criterion = torch.nn.CrossEntropyLoss()

# calculate the euclidean distance and global structure between models
print("Calculating the euclidean distance between models...")
X_transformed, global_structure = operation.calculate_model_similarity_global_structure(dataset_name, subdataset_list, IN_DIM, OUT_DIM)
print("Finished calculating the euclidean distance between models.")

# print the euclidean distance matrix and the global structure
print("Euclidean distance between models: ", X_transformed)
print("Euclidean distance matrix shape: ", X_transformed.shape)

# save the model euclidean distance similarity results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'modelMDS': X_transformed.tolist()}
record_id = operation.store_into_database(client, database_name, collection_name_model_euclidean_distance_similarity, record)
print("One record of model_euclidean_distance_similarity has been successfully inserted with ID " + str(record_id))

# calculate the cka distance between models
print("Calculating the CKA similarity between models...")
reshape_x = int(args.reshapex)
reshape_y = int(args.reshapey)
linear_cka_embedding, kernel_cka_embedding, linear_cka_figure, kernel_cka_figure = operation.calculate_model_cka_similarity_global_structure(dataset_name, subdataset_list, IN_DIM, OUT_DIM, reshape_x, reshape_y)
print("Finished calculating the CKA similarity between models.")

# print the CKA distances matrix and the MDS results
print("Linear CKA similarity distance between models: ", linear_cka_embedding)
print("Linear CKA similarity matrix shape: ", linear_cka_embedding.shape)
print("RBF kernel CKA similarity distance between models: ", kernel_cka_embedding)
print("RBF kernel CKA similarity matrix shape: ", kernel_cka_embedding.shape)

# save the model CKA similarity results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'linearCKA_MDS': linear_cka_embedding.tolist(), 'kernelCKA_MDS': kernel_cka_embedding.tolist()}
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

# calculate the euclidean distance between model layer weights
print("Calculating the euclidean distance between model weights...")
X_transformed_layerOne, X_transformed_layerTwo, matrix_LayerOne, matrix_LayerTwo = operation.calculate_layer_euclidean_distance_similarity(dataset_name, subdataset_list, IN_DIM, OUT_DIM)
print("Finished calculating the euclidean distance between model weights.")

# print the layer euclidean distance matrix
print("Euclidean distance among LayerOne: ", X_transformed_layerOne)
print("Euclidean distance LayerOne shape: ", X_transformed_layerOne.shape)
print("Euclidean distance among LayerTwo: ", X_transformed_layerTwo)
print("Euclidean distance LayerTwo shape: ", X_transformed_layerTwo.shape)

# save the model weight euclidean distance similarity results to the database
record = {'dataset': dataset_name, 'model_type': model_name, 'layerOneMDS': X_transformed_layerOne.tolist(), 'layerTwoMDS': X_transformed_layerTwo.tolist(), 'layerOneMatrix': matrix_LayerOne.tolist(), 'layerTwoMatrix': matrix_LayerTwo.tolist()}
record_id = operation.store_into_database(client, database_name, collection_name_layer_euclidean_distance_similarity, record)
print("One record of layer_euclidean_distance_similarity has been successfully inserted with ID " + str(record_id))
print("The model layer euclidean distance similarity information have been saved into MongoDB.")

# # calculate the torch CKA similarity between layers
# print("Calculating the CKA similarity between model layers...")
# cka_result_all = operation.calculate_model_layer_torch_cka_similarity(dataset_name, subdataset_list, IN_DIM, OUT_DIM)
# print("Finished calculating the CKA similarity between model layers.")

# # print the torch CKA similarity between models
# for i in range(len(cka_result_all)):
#     cka_result = cka_result_all[i]
#     for j in range(len(cka_result)):
#         print("Torch CKA similarity between model " + str(i) + " and model " + str(j) + " is: ", cka_result[j]["CKA"])

# # save the torch CKA similarity into database
# for i in range(len(cka_result_all)):
#     cka_result = cka_result_all[i]
#     for j in range(len(cka_result)):
#         record = {'dataset': dataset_name, 'model_type': model_name, 'model1': str(i), 'model2': str(j), 'torchCKA_similarity': cka_result[j]["CKA"].tolist()}
#         record_id = operation.store_into_database(client, database_name, collection_name_layer_torch_cka_similarity, record)
#         print("One record of layer_torch_cka_similarity has been successfully inserted with ID " + str(record_id))
# print("Layer torch CKA similarity information have been saved into MongoDB.")

# calculate the torch CKA similarity between layers
print("Calculating the CKA similarity between model layers...")
cka_result_all = operation.calculate_model_layer_torch_cka_similarity(dataset_name, subdataset_list, IN_DIM, OUT_DIM)
print("Finished calculating the CKA similarity between model layers.")

# print the torch CKA similarity between models
for i in range(len(cka_result_all)):
    cka_result = cka_result_all[i]
    for j in range(len(cka_result)):
        print("Torch CKA similarity between model " + str(i) + " and model " + str(j) + " is: ", cka_result[j]["CKA"])

# save the torch CKA similarity into database
for i in range(len(cka_result_all)):
    cka_result = cka_result_all[i]
    for j in range(len(cka_result)):
        record = {'dataset': dataset_name, 'model_type': model_name, 'model1': str(i), 'model2': str(j), 'torchCKA_similarity': cka_result[j]["CKA"].tolist()}
        record_id = operation.store_into_database(client, database_name, collection_name_layer_torch_cka_similarity, record)
        print("One record of layer_torch_cka_similarity has been successfully inserted with ID " + str(record_id))
print("Layer torch CKA similarity information have been saved into MongoDB.")

# calculate the top eigenvalues for hessian
print("Calculating the top eigenvalues for hessian...")
top_eigenvalues_list = operation.calculate_top_eigenvalues_hessian(dataset_name, subdataset_list, criterion, IN_DIM, OUT_DIM)
print("Finished calculating the top eigenvalues for hessian.")

# # calculate the hessian loss landscape and model information of the perb model
# for k in tqdm(range(len(subdataset_list)), desc="Calculating the hessian loss landscape and model information"):
#     top_eigenvalues, result_string, result_array, model_list, model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix = operation.calculate_hessian_loss_landscape(dataset_name, subdataset_list, k, criterion, x, y, IN_DIM, OUT_DIM, STEPS, START, END)
    
#     # save the results to the database
#     record = {"dataset": dataset_name, 'model_type': model_name, "testing": "all", "training": subdataset_list[k], "steps": STEPS, "start": START, "end": END, "top_eigenvalues": top_eigenvalues, "result": result_string}
#     record_id = operation.store_into_database(client, database_name, collection_name_hessian_contour, record)
#     # print("One record of hessian_contour has been successfully inserted with ID " + str(record_id))

#     # save the loss landscapes for hessian results to the database
#     record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[k], "steps": STEPS, "loss_data_fin": result_array.tolist()}
#     record_id = operation.store_into_database(client, database_name, collection_name_hessian_loss_landscape, record)
#     # print("One record of hessian_loss_landscape has been successfully inserted with ID " + str(record_id))
    
#     # save the model information to the database
#     for i in range(STEPS):
#         for j in range(STEPS):
#             # print("i: " + str(i) + " j: " + str(j))
#             record = {'dataset': dataset_name, 'model_type': model_name, 'testing': "all",'training': subdataset_list[k],'step1':i,'step2':j, 'accuracy':model_accuracy[i*STEPS+j].tolist(),'recall':model_recall[i*STEPS+j].tolist(),'precision':model_precision[i*STEPS+j].tolist(),'f1':model_f1[i*STEPS+j].tolist(),'confusionMatrix':model_confusionMatrix[i*STEPS+j].tolist()}
#             record_id = operation.store_into_database(client, database_name, collection_name_hessian_contour_model_information, record)
#             # print("One record of hessian_contour_model_information has been successfully inserted with ID " + str(record_id))

# print("All the hessian contour models with their information have been saved into MongoDB.")

# calculate the training loss landscape
print("Calculating the training loss landscape...")
loss_data_fin_list_train, model_info_list_train, max_loss_value_list_train, min_loss_value_list_train = operation.calculate_training_loss_landscape(dataset_name, subdataset_list, criterion, IN_DIM, OUT_DIM, LOSS_STEPS)
print("Finished calculating the training loss landscape.")

# save the single model results to the database
for i in range(len(loss_data_fin_list_train)):
    record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS, "loss_data_fin": loss_data_fin_list_train[i].tolist(), "max_loss_value_x": max_loss_value_list_train[2*i].tolist(), "max_loss_value_y": max_loss_value_list_train[2*i+1].tolist(), "min_loss_value_x": min_loss_value_list_train[2*i].tolist(), "min_loss_value_y": min_loss_value_list_train[2*i+1].tolist()}
    record_id = operation.store_into_database(client, database_name, collection_name_training_loss_landscape, record)
    print("One record of training_loss_landscape has been successfully inserted with ID " + str(record_id))
print("All the training loss landscape models have been saved into MongoDB.")

# prepare testing data used for evaluating testing loss landscape
print("Loading MNIST and MNIST-C dataset for evaluating loss...")
# original data used for evaluating loss
mnist_original = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
mnist_original_train_loader = torch.utils.data.DataLoader(mnist_original, batch_size=1, shuffle=False)
x, y = iter(mnist_original_train_loader).__next__()
mnist_test = []
for i in range(len(x)):
    mnist_test.append([x[i], y[i]])
x_c = load('../data/MNIST_C/brightness/train_images.npy')
y_c = load('../data/MNIST_C/brightness/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/canny_edges/train_images.npy')
y_c = load('../data/MNIST_C/canny_edges/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/dotted_line/train_images.npy')
y_c = load('../data/MNIST_C/dotted_line/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/fog/train_images.npy')
y_c = load('../data/MNIST_C/fog/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/glass_blur/train_images.npy')
y_c = load('../data/MNIST_C/glass_blur/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/impulse_noise/train_images.npy')
y_c = load('../data/MNIST_C/impulse_noise/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/motion_blur/train_images.npy')
y_c = load('../data/MNIST_C/motion_blur/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/rotate/train_images.npy')
y_c = load('../data/MNIST_C/rotate/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/scale/train_images.npy')
y_c = load('../data/MNIST_C/scale/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/shear/train_images.npy')
y_c = load('../data/MNIST_C/shear/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/shot_noise/train_images.npy')
y_c = load('../data/MNIST_C/shot_noise/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/spatter/train_images.npy')
y_c = load('../data/MNIST_C/spatter/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/stripe/train_images.npy')
y_c = load('../data/MNIST_C/stripe/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/translate/train_images.npy')
y_c = load('../data/MNIST_C/translate/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
x_c = load('../data/MNIST_C/zigzag/train_images.npy')
y_c = load('../data/MNIST_C/zigzag/train_labels.npy')
x_c = torch.from_numpy(x_c.reshape(60000,784)).float()
y_c = torch.from_numpy(y_c.reshape(60000)).long()
for i in range(len(x_c)):
    mnist_test.append([x_c[i], y_c[i]])
# define testing data loader
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

# set x and y from test_loader
x, y = iter(test_loader).__next__()

# calculate the loss landscapes for random projection
print("Calculating the loss landscapes for random projection...")
loss_data_fin_list, model_info_list, max_loss_value_list, min_loss_value_list = operation.calculate_loss_landscapes_random_projection(dataset_name, subdataset_list, criterion, x, y, IN_DIM, OUT_DIM, LOSS_STEPS)
print("The loss landscapes for random projection have been successfully calculated.")

# calculate and save the model analysis information
print("Calculating the model analysis information...")
for i in range(len(subdataset_list)):
    model_accuracy, model_recall, model_precision, model_f1, model_confusionMatrix = operation.calculate_model_information(dataset_name, subdataset_list, i, x, y, IN_DIM, OUT_DIM)
    record = {'dataset': dataset_name, 'model_type': model_name, 'model': subdataset_list[i],'accuracy':model_accuracy.tolist(),'recall':model_recall.tolist(),'precision':model_precision.tolist(),'f1':model_f1.tolist(),'confusionMatrix':model_confusionMatrix.tolist(), 'top_eigenvalues': top_eigenvalues_list[i]}
    record_id = operation.store_into_database(client, database_name, collection_name_model_analysis_information, record)
    print("One record of model_analysis_information has been successfully inserted with ID " + str(record_id))
print("Finished calculating the model analysis information.")

# prepare the calculation parameters
DETAILED_START = int(5)
DETAILED_END = int(LOSS_STEPS / 2)
DETAILED_STEP = int(5)
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

# transfer the model parameters into models
print("Transfer the model parameters into models...")
selected_model_list_in_model_shape_all_steps = []
for i in range(len(selected_model_list_all_steps)):
    selected_model_list_in_model_shape_all_steps.append(operation.transfer_model_parameters_into_models(selected_model_list_all_steps[i], IN_DIM, OUT_DIM))
print("Finish transfer the model parameters into models.")

# save the single model results to the database
for i in range(len(loss_data_fin_list)):
    one_record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS, "loss_data_fin": loss_data_fin_list[i].tolist(), "max_loss_value_x": max_loss_value_list[2*i].tolist(), "max_loss_value_y": max_loss_value_list[2*i+1].tolist(), "min_loss_value_x": min_loss_value_list[2*i].tolist(), "min_loss_value_y": min_loss_value_list[2*i+1].tolist()}
    record.update({subdataset_list[i]: one_record})
    record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS, "loss_data_fin": loss_data_fin_list[i].tolist(), "max_loss_value_x": max_loss_value_list[2*i].tolist(), "max_loss_value_y": max_loss_value_list[2*i+1].tolist(), "min_loss_value_x": min_loss_value_list[2*i].tolist(), "min_loss_value_y": min_loss_value_list[2*i+1].tolist()}
    record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_contour, record)
    print("One record of loss_landscapes_contour has been successfully inserted with ID " + str(record_id))
print("Finished saving the single model results to the database.")

# save the global result to the database
loss_data_fin_array = np.array(loss_data_fin_list)
record = {"dataset": dataset_name, 'model_type': model_name, "steps": LOSS_STEPS, "loss_data_fin": loss_data_fin_array.tolist()}
record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_global, record)
print("One record of loss_landscapes_global has been successfully inserted with ID " + str(record_id))

# generate training 3D loss landscapes for random projection
print("Calculating the training 3D loss landscapes for random projection...")
loss_data_fin_list_3d_training = operation.calculate_training_3d_loss_landscapes_random_projection(dataset_name, subdataset_list, criterion, IN_DIM, OUT_DIM, LOSS_STEPS)
print("Finished calculating the training 3D loss landscapes for random projection.")

# save the results to the database
for i in range(len(loss_data_fin_list_3d_training)):
    record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS, "loss_data_fin_3d": loss_data_fin_list_3d_training[i].tolist()}
    record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_contour_3d_training, record)
    print("One record of loss_landscapes_contour_3d_training has been successfully inserted with ID " + str(record_id))
print("Finished saving the loss_landscapes_contour_3d_training results to the database.")

# prepare the data to store in .vti files for ttk input
nx, ny, nz = LOSS_STEPS - 1, LOSS_STEPS - 1, LOSS_STEPS - 1
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Variables
pressure = np.zeros(ncells).reshape( (nx, ny, nz), order = 'C')
# temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1))

# store the loss landscape results into binary files used for ttk
for i in range(len(loss_data_fin_list_3d_training)):
    FILE_PATH = '../ttk/input_binary_for_ttk/MNIST_' + model_name + '_' + subdataset_list[i] + '_loss_mnist_training_3d_contour'
    imageToVTK(FILE_PATH, cellData = {"Cell" : pressure}, pointData = {"Loss" : loss_data_fin_list_3d_training[i]})

# generate testing 3D loss landscapes for random projection
print("Calculating the 3D loss landscapes for random projection...")
loss_data_fin_list_3d = operation.calculate_3d_loss_landscapes_random_projection(dataset_name, subdataset_list, criterion, x, y, IN_DIM, OUT_DIM, LOSS_STEPS)
print("The 3D loss landscapes for random projection have been successfully calculated.")

# save the results to the database
for i in range(len(loss_data_fin_list_3d)):
    record = {"dataset": dataset_name, 'model_type': model_name, "model": subdataset_list[i], "steps": LOSS_STEPS, "loss_data_fin_3d": loss_data_fin_list_3d[i].tolist()}
    record_id = operation.store_into_database(client, database_name, collection_name_loss_landscapes_contour_3d, record)
    print("One record of loss_landscapes_contour_3d has been successfully inserted with ID " + str(record_id))
print("Finished saving the loss_landscapes_contour_3d results to the database.")

# # prepare for the binary storage to store as .vtr files for ttk input
# loss_3d_x = np.arange(0, LOSS_STEPS)
# loss_3d_y = np.arange(0, LOSS_STEPS)
# loss_3d_z = np.arange(0, LOSS_STEPS)

# prepare the data to store in .vti files for ttk input
nx, ny, nz = LOSS_STEPS - 1, LOSS_STEPS - 1, LOSS_STEPS - 1
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Variables
pressure = np.zeros(ncells).reshape( (nx, ny, nz), order = 'C')
# temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1))

# store the loss landscape results into binary files used for ttk
for i in range(len(loss_data_fin_list_3d)):
    FILE_PATH = '../ttk/input_binary_for_ttk/MNIST_' + model_name + '_' + subdataset_list[i] + '_loss_mnist_3d_contour'
    # np.save('../ttk/input_binary_for_ttk/MNIST_' + model_name + '_' + subdataset_list[i] + '_loss_mnist_3d_contour.npy', loss_data_fin_list_3d[i])
    # gridToVTK(FILE_PATH, loss_3d_x, loss_3d_y, loss_3d_z, cellData = {'Loss': loss_data_fin_list_3d[i]})
    imageToVTK(FILE_PATH, cellData = {"Cell" : pressure}, pointData = {"Loss" : loss_data_fin_list_3d[i]})

# prepare the testing dataset for prediction distribution computation
print("Loading MNIST and MNIST-C dataset for prediction distribution...")
# original data used for prediction distribution
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
# define testing data loader for prediction distribution calculation
distribution_test_loader = torch.utils.data.DataLoader(mnist_test_prediction_distribution, batch_size=1, shuffle=False)
print("Finish loading the testing data for prediction distribution calculation.")

# calculate the prediction distribution of all models
print("Calculating the prediction distribution of all models...")
# select two model sets
for k in range(DETAILED_START, DETAILED_END, DETAILED_STEP):
    index = int(k/DETAILED_STEP) - 1
    selected_model_list_in_model_shape_all = selected_model_list_in_model_shape_all_steps[index]
    for i in range(len(selected_model_list_in_model_shape_all)):
        for j in range(len(selected_model_list_in_model_shape_all)):
            # calculate the prediction distribution
            correct_correct, correct_wrong, wrong_correct, wrong_wrong = operation.calculate_prediction_distribution(selected_model_list_in_model_shape_all[i], selected_model_list_in_model_shape_all[j], distribution_test_loader, MAX_NUM)
            print("Finish calculating the prediction distribution of model set " + str(i) + " and model set " + str(j) + ".")
            # save the prediction distribution to the database
            record = {"dataset": dataset_name, 'model_type': model_name, "model1": subdataset_list[i], "model2": subdataset_list[j], "correct_correct": correct_correct, "correct_wrong": correct_wrong, "wrong_correct": wrong_correct, "wrong_wrong": wrong_wrong, "threshold": k}
            record_id = operation.store_into_database(client, database_name, collection_name_prediction_distribution, record)
            print("One record of prediction_distribution has been successfully inserted with ID " + str(record_id))
print("Finish calculating the prediction distribution of all models.")