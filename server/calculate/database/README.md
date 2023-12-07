# Database Instructions
This is an instruction to introduce all the database information in this tool.

## Database Collection Format
This section will introduce the format of the MongoDB database. A record in MongoDB is a document, which is a data structure composed of field and value pairs. MongoDB documents are similar to JSON objects. The values of fields may include other documents, arrays and arrays of documents.

In this project, we will store different figure data in different collections. We will introduce those collections one by one.

### hessian_contour
`hessian_contour` collection contains all the hessian coutour figure data of different models with different testing data. The figure data of the hessian contour is a 2D matrix based on top first and second hessian eigenvectors. The function to use the hessian eigenvector to modify the model is as follows.

```python
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb
```

The core code to calculate the hessian contour is as follows.

```python
# calculate hessian contour
hessian_comp = hessian(model_final, criterion, data=(x, y), cuda=False)
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
lams = np.linspace(-0.5, 0.5, STEPS).astype(np.float32)
loss_list = []
for lam in lams:
    model_perb = get_params(model_initial, model_final, top_eigenvector[0], lam)
    loss_list_sec = []
    for lam_sec in lams:
        model_perb_sec = get_params(model_initial, model_perb, top_eigenvector[1], lam_sec)
        loss_list_sec.append(criterion(model_perb_sec(x), y).item())
    loss_list.append(loss_list_sec)
```

The `STEPS` will determine the 2D matrix size. (For example, if the `STEPS` is set as `25`, the 2D hessian contour result matrix would be `25*25`.) To save this result in the MongoDB, we will create one document(record) in the following way.

```python
record = {"dataset": args.dataset, "testing": args.testing, "training": args.training, "steps": str(STEPS), "result": result_string}
```

One example document of `hessian_contour` is shown as follows.

```JSON
{
  "dataset": "MNIST",
  "result": "2.3405797481536865,2.3318138122558594,2.3279941082000732,...,2.309490442276001,2.312577486038208,2.3185887336730957",
  "steps": "25",
  "testing": "original",
  "training": "original"
}
```

The `"result"` contains `25*25` numbers in total as the steps is set as `25`.

### model information (including weight and bias)
To generate the basic information of a model which can be used for the visualization in the Front-End, you need to run the `mnist_model_info.py` file under the `calculate` folder.

```shell
python3 mnist_model_info.py --dataset=<training and testing dataset> --testing=<testing sub-dataset> ---training=<training sub-dataset>
```

The training and testing dataset options are `MNIST` or `CIFAR10`.

The testing sub-dataset options are `original`, `brightness`, `canny_edges`, `dotted_line`, `fog`, `glass_blur`, `identity`, `impulse_noise`, `motion_blur`, `rotate`, `scale`, `shear`, `shot_noise`, `spatter`, `stripe`, `translate`, `zigzag`, and `all`. The default setting is `original`.

The training sub-dataset options are `original`, `brightness`, `canny_edges`, `dotted_line`, `fog`, `glass_blur`, `identity`, `impulse_noise`, `motion_blur`, `rotate`, `scale`, `shear`, `shot_noise`, `spatter`, `stripe`, `translate`, `zigzag` and `all`. The default setting is `original`.

We do provide a script file as well for `mnist_model_info.py` which contains all the commands to obtain and store all the results to MongoDB for all information including the weights and bias with `original` MNIST and MNIST-C. This script will generate `49` results in total and store all of them in MongoDB. You could just run the following command to finish calculating and putting the results into MongoDB.

```shell
sh run_mnist_model_info.sh
```

You can modify the script file to get comparion results between other sub-datasets. All the results will be saved in the MongoDB database. Once the server has started and received requests from the Front-End, `server.py` will return corresponding results.

The `STEPS` will determine the number of results for each document(record). (For example, if the `STEPS` is set as `25`, the total number of saved model will be `25`. Each model will have its weight and bias for each layer.) To save this result in the MongoDB, we will create one document(record) in the following way.

```python
record = {'dataset': args.dataset,'testing': args.testing,'training': args.training,'step1':i,'step2':j,'model_layer1_weight':model_layer1_weight[i*STEPS+j].tolist(),'model_layer1_bias':model_layer1_bias[i*STEPS+j].tolist(),'model_layer2_weight':model_layer2_weight[i*STEPS+j].tolist(),'model_layer2_bias':model_layer2_bias[i*STEPS+j].tolist(),'accuracy':model_accuracy[i*STEPS+j].tolist(),'recall':model_recall[i*STEPS+j].tolist(),'precision':model_precision[i*STEPS+j].tolist(),'f1':model_f1[i*STEPS+j].tolist(),'confusionMatrix':model_confusionMatrix[i*STEPS+j].tolist()}
```

One example document of `hessian_contour_model_weight_bias` is shown as follows.

```JSON
{
    {
  "dataset": "MNIST",
  "model_layer1_bias": "...",
  "model_layer1_weight": "...",
  "model_layer2_bias": "...",
  "model_layer2_weight": "...",
  "step1": 0,
  "step2": 0,
  "testing": "original",
  "training": "original"
}
}
```

### model merge tree
To generate the merge tree of a model which can be used for the visualization in the Front-End, you need to run the `load_from_csv.py` file under the `calculate` folder.

```shell
python3 load_from_csv.py --file=<input csv file obtained from ttk>
```

We do provide a script file as well for `load_from_csv.py` which contains all the commands to obtain and store all the results to MongoDB for given input csv files generated by `ttk`.

```shell
sh run_load_from_csv.sh
```

You can modify the script file to save other results of ttk output csv files in the MongoDB database. Once the server has started and received requests from the Front-End, `server.py` will return corresponding results.

## MongoDB Operations
We have provided some operations from Python to deal with the MongoDB database such as delete documents or drop collections. You could find more functions(operations) in the `functions.py` under current folder.

### Drop Collections
The functions to drop a collection is defined as follows.

```python
def drop(client, database_name, collection_name):
    myclient = pymongo.MongoClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    mycol.drop()
    print("Successfully drop one collection.")
    return True
```

You could use this function by calling the `drop_collection.py` under the `calculate` folder in the following way:

```shell
python3 drop_collection.py --client=<client port> --database=<database name> --collection=<collection name>
```

It is highly recommanded to add all the options while using this Python file. The default value of those options are as follows.

```python
# client port
parser.add_argument('--client', default='mongodb://localhost:27017/', help='client port')
# database name
parser.add_argument('--database', default='mydatabase', help='database name')
# collection name
parser.add_argument('--collection', default='hessian_contour', help='collection name')
```

### Delete Documents
We have provided functions to delete one or many documents(records) in a given collection. The functions are defined as follows.

```python
def deleteOne(client, database_name, collection_name, query):
    myclient = pymongo.MongoClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    mycol.delete_one(query)
    print("Successfully delete one document.")
    return True

def deleteMany(client, database_name, collection_name, query):
    myclient = pymongo.MongoClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    x = mycol.delete_many(query)
    print(x.deleted_count, " documents have been successfully deleted.")
    return True
```

You could use this function by calling the `delete_document.py` under the `calculate` folder in the following way:

```shell
python3 delete_document.py --client=<client port> --database=<database name> --collection=<collection name> --query=<delete query> --delete_option=<delete option>
```

It is highly recommanded to add all the options while using this Python file. The default value of those options are as follows.

```python
# client port
parser.add_argument('--client', default='mongodb://localhost:27017/', help='client port')
# database name
parser.add_argument('--database', default='mydatabase', help='database name')
# collection name
parser.add_argument('--collection', default='hessian_contour', help='collection name')
# query
parser.add_argument('--query', default='{}', help='query')
# delete_option
parser.add_argument('--delete_option', default='one', help='one/many documents to be deleted')
```