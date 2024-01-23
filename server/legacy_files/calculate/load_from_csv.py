import csv
import argparse
from database import functions

# load the parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--file', default='resnet56_noshort_sgd.csv', help='input csv file')
parser.add_argument('--modelType', default='resnet56', help='model type')
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
file_path = '../ttk/output_csv_from_ttk/' + args.file
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

# write the data to database
client = "mongodb://localhost:27017/"
database_name = "mydatabase"
collection_name = "ttk_merge_tree"

if(functions.empty(client,database_name,collection_name) == False):
    print("The " + collection_name + " collection exists.")
else:
    print("The " + collection_name + " collection does not exist.")
    print("Create the new collection.")
    functions.create(client,database_name,collection_name)

for i in range(len(pointsx)):
    record_id = functions.insert(client,database_name,collection_name,{'model_type': args.modelType, 'model':args.file, 'start':start[i], 'end':end[i], 'x':pointsx[i], 'y':pointsy[i], 'z':pointsz[i], 'nodeID':nodeID[i],'branchID':branchID[i]})
    print("One record of ttk_merge_tree has been successfully inserted with ID " + str(record_id))
print("All the merge tree information have been saved into MongoDB.")
