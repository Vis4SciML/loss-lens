import csv
import argparse
from database import functions

# load the parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="local", help='device to do the back-end calculation')
parser.add_argument('--file', default='../ttk/output_csv_from_ttk/MNIST_MergeTree3D_Training/MNIST_CNN_all_loss_mnist_training_3d_contour.csv', help='input csv file')
args = parser.parse_args()

# lists used to store the data
pointsx = []
pointsy = []
pointsz = []
nodeID = []
branchID = []
start = []
end = []

# initialize some global variables
root_x = 0

# open the csv file and load the data into the lists
file_path = args.file
input_information_list = file_path.split('/')
input_information = input_information_list[-1]
dataset = input_information.split('_')[0]
model_type = input_information.split('_')[1]
model = input_information.split('_')[2]
print('dataset:', dataset, 'model_type:', model_type, 'model:', model)

with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pointsx.append(float(row['Points_0']))
        pointsy.append(float(row['Points_1']))
        pointsz.append(float(row['Points_2']))
        nodeID.append(int(row['Point ID']))
        branchID.append(int(row['BranchNodeID']))
        # find the start point of each branch
        if int(row['BranchNodeID']) == 0:
            if int(row['Point ID']) == 0:
                root_x = float(row['Points_0'])
                start.append(1)
                end.append(0)
            else:
                start.append(0)
                end.append(1)
        else:
            if float(row['Points_0']) == root_x:
                start.append(1)
                end.append(0)
            else:
                start.append(0)
                end.append(0)

# find the end point of each branch
for i in range(len(start)):
    this_x = pointsx[i]
    this_y = pointsy[i]
    this_z = pointsz[i]
    for j in range(len(start)):
        if this_x == pointsx[j] and this_y == pointsy[j] and this_z == pointsz[j] and i != j:
            end[i] = 1
            end[j] = 1

# verify that the start and end points are correct
for i in range(len(start)):
    print('start: ', start[i], 'end: ', end[i], 'x: ', pointsx[i], 'y: ', pointsy[i], 'z: ', pointsz[i], 'nodeID: ', nodeID[i], 'branchID: ', branchID[i])

# set the client and database name
client = "mongodb://localhost:27017/"
database_name = "mydatabase"
if args.device == "nersc":
    client = "mongodb07.nersc.gov"
    database_name = "losslensdb"

# set the collection name
collection_name = "ttk_merge_tree_training"

if(functions.empty(client,database_name,collection_name) == False):
    print("The " + collection_name + " collection exists.")
else:
    print("The " + collection_name + " collection does not exist.")
    print("Create the new collection.")
    functions.create(client,database_name,collection_name)

for i in range(len(pointsx)):
    record_id = functions.insert(client,database_name,collection_name,{'dataset': dataset, 'model_type':model_type, 'model':model, 'start':start[i], 'end':end[i], 'x':pointsx[i], 'y':pointsy[i], 'z':pointsz[i], 'nodeID':nodeID[i],'branchID':branchID[i]})
    print("One record of ttk_merge_tree has been successfully inserted with ID " + str(record_id))
print("All the merge tree information have been saved into MongoDB.")
