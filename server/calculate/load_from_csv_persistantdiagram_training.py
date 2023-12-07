import csv
import argparse
from database import functions

# load the parameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="local", help='device to do the back-end calculation')
parser.add_argument("--file", default="../ttk/output_csv_from_ttk/MNIST_PersistantDiagram_Training/MNIST_CNN_all_loss_mnist_training_3d_contour.csv", help="input csv file")
args = parser.parse_args()

# lists used to store the data
points_0 = []
points_1 = []
points_2 = []
nodeID = []

# open the csv file and load the data into the lists
file_path = args.file
input_information_list = file_path.split("/")
input_information = input_information_list[-1]
dataset = input_information.split("_")[0]
model_type = input_information.split("_")[1]
model = input_information.split("_")[2]
print("dataset:", dataset, "model_type:", model_type, "model:", model)

# load coordinates from csv file
with open(file_path, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        points_0.append(float(row["Points_0"]))
        points_1.append(float(row["Points_1"]))
        points_2.append(float(row["Points_2"]))
        nodeID.append(int(row["Point ID"]))

# verify the data
for i in range(len(nodeID)):
    print(
        "nodeID:",
        nodeID[i],
        "coordinates_0:",
        points_0[i],
        "coordinates_1:",
        points_1[i],
        "coordinates_2:",
        points_2[i],
    )

# set the client and database name
client = "mongodb://localhost:27017/"
database_name = "mydatabase"
if args.device == "nersc":
    client = "mongodb07.nersc.gov"
    database_name = "losslensdb"

# set the collection name
collection_name = "ttk_persistant_diagram_training"

if functions.empty(client, database_name, collection_name) == False:
    print("The " + collection_name + " collection exists.")
else:
    print("The " + collection_name + " collection does not exist.")
    print("Create the new collection.")
    functions.create(client, database_name, collection_name)

for i in range(len(nodeID)):
    record_id = functions.insert(
        client,
        database_name,
        collection_name,
        {
            "dataset": dataset,
            "model_type": model_type,
            "model": model,
            "coordinates_0": points_0[i],
            "coordinates_1": points_1[i],
            "coordinates_2": points_2[i],
            "nodeID": nodeID[i],
        },
    )
    print(
        "One record of ttk_persistant_diagram has been successfully inserted with ID "
        + str(record_id)
    )
print("All the ttk persistant diagram information have been saved into MongoDB.")
