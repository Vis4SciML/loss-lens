# Loss Lens
This project provides a fullstack tool for loss visualization to help machine learning experts to understand how the augmented data improves the model.

## Setup Docker
To set up Docker, run the following under the root directory:
```
docker-compose up
```

## Front-End Instructions
We give a list of front-end instructions as follows in this section, including some installation instructions of Node.js, Yarn and other dependencies. We also have one instruction of how to run the front-end.
### [Install Node.js](frontend/README.md#install-nodejs)
### [Install Yarn](frontend/README.md#install-yarn)
### [Install Dependencies](frontend/README.md#install-dependencies)
### [Start the Front-End](frontend/README.md#run)

## Back-End Instructions
We give a list of back-end instructions as follows in this section, including Package Install Instructions, MongoDB(database used in this tool) Set-Up Instructions, Training model Instructions, Calculation for Visualization Instructions, Database Instructions and the way to Start the Back-End Server.
### [Package Install Instructions](server/README.md#package-install-instructions)
### [MongoDB Set-Up Instructions](server/README.md#mongodb-set-up-instructions)
### [Training model Instructions](server/train/README.md#training-model-instructions)
### [Calculation for Visualization Instructions](server/calculate/README.md#server-pipeline-instructions)
### [Database Instructions](server/calculate/database/README.md#database-instructions)
### [Start the Back-End Server](server/README.md#start-the-back-end-server)

## Simple Example
Here, we provide a simple example to show how to use this tool for loss visualization. We take the 2-Layers MLP with MNIST as an example.

### Prepare Pre-Trained Models for Comparison
The first step to visualize several models in this tool is to prepare several pre-trained models for comparison. We provide some examples under the [train](server/train/) folder. You can also make your own training code under the [train](server/train/) folder or you can directly put your own pre-trained models under the [model](server/model/) folder. For training models, you can refer the [Training model Instructions](server/train/README.md#training-model-instructions).

Here, we take the [2-Layers MLP with MNIST](server/train/README.md#2-layers-mlp-with-mnist) as an example. To train the models, you can run the following commands under the [train](server/train/) folder.

```shell
sh run_train_mlp_mnist.sh
```

One `model_initial.pt` and one `model_final.pt` will be generated under the corresponding folder which should be under `model/MNIST` folder. For example, two models of `original` training option will be generated under `model/MNIST/original` folder.

### Run the Calculations
As all the calculation results will be saved into the MongoDB database, one step before running the calculation is to start the MongoDB. You can find more details in the [MongoDB Set-Up Instructions](server/README.md#mongodb-set-up-instructions).

In order to visualize the loss from the Front-End, after we have the models, we need to calculate all the results the Front-End will need for visualization purpose. All the related functions are in the [functions.py](server/calculate/operation/functions.py) and you can find more details in the [Back-End Operations Instructions](server/calculate/operation/README.md#back-end-operations-instructions).

Here, we take the [2-Layers MLP with MNIST](server/train/README.md#2-layers-mlp-with-mnist) as an example. To run the calculations, you can run the following commands under the [calculate](server/calculate/) folder.

```shell
python3 calculate_mlp_mnist.py
```

All the results obtained from the above steps will be saved in the MongoDB and all the generated model files will be saved under the `model_list` folder and the generated binary files used for ttk will be saved under the `ttk/input_binary_for_ttk` folder.

You need to do two more steps to finish all the Back-End calculations, which is related to the [TTK](https://topology-tool-kit.github.io/). We will generated the binary files under the `ttk/input_binary_for_ttk/` folder while running the `calculate_mlp_mnist.py`. `TTK` will take the binary files to generate a CSV file in the [ParaView](https://www.paraview.org/). Here, we will show how to obtain meaningful information from the output CSV file and store the information which Front-End might need in the MongoDB. The output CSV files are under `ttk/output_csv_from_ttk/` folder. To obtain meaningful information, we need to run `load_from_csv_mergetree.py` by giving a CSV file path.

```shell
python3 load_from_csv_mergetree.py --file=<input CSV file path>
```

A script file which contains several example commands can be run as following.

```shell
sh run_load_from_csv_mergetree.sh
```

All the results will be saved in the MongoDB. `load_from_csv_mergetree_nersc.py` and `run_load_from_csv_mergetree_nersc.sh` are designed for running this script in `NERSC`. `load_from_csv_mergetree_nersc_training.py` and `run_load_from_csv_mergetree_nersc_training.sh` are designed for running the 3D loss landscapes merge tree in `NERSC`.

Another visualization in the 3D loss landscapes is the `Persistant Diagram`, which is also generated by the [TTK](https://topology-tool-kit.github.io/). We will generated the binary files under the `ttk/input_binary_for_ttk/` folder while running the `calculate_mlp_mnist.py`. `TTK` will take the binary files to generate a CSV file in the [ParaView](https://www.paraview.org/). Here, we will show how to obtain meaningful information from the output CSV file and store the information which Front-End might need in the MongoDB. The output CSV files are under `ttk/output_csv_from_ttk/` folder. To obtain meaningful information, we need to run `load_from_csv_persistantdiagram.py` by giving a CSV file path.

```shell
python3 load_from_csv_persistantdiagram.py --file=<input CSV file path>
```

A script file which contains several example commands can be run as following.

```shell
sh run_load_from_csv_persistantdiagram.sh
```

All the results will be saved in the MongoDB. `load_from_csv_persistantdiagram_nersc.py` and `run_load_from_csv_persistantdiagram_nersc.sh` are designed for running this script in `NERSC`. `load_from_csv_persistantdiagram_nersc_training.py` and `run_load_from_csv_persistantdiagram_nersc_training.sh` are designed for running the 3D loss landscapes merge tree in `NERSC`.

### Visualize from the Front-End
After you finish all the calculations, you need to [Start the Back-End Server](server/README.md#start-the-back-end-server) and also [Start the Front-End](frontend/README.md#run).

## Known Issues
There are several known issues when running this tool.

### RobustBench Package MacOS Issue
If you see a error of this sorts:

```shell
Traceback (most recent call last):
  File ***, line ***, in <module>
    from robustbench.data import load_cifar10c
  File ***, line ***, in <module>
    from .eval import benchmark
  File ***, line ***, in <module>
    from autoattack.state import EvaluationState
ModuleNotFoundError: No module named 'autoattack.state'
```

This is one error related to the `robustbench.data` package from the [RobustBench](https://github.com/RobustBench/robustbench) in MacOS. This kind of errors will not appear in our Ubuntu environment. The suggested environment is Ubuntu.

### MongoDB Issue
If you see a error of this sorts:

```shell
  File ***, line ***, in _get_socket
    sock_info = self.connect(handler=handler)
  File ***, line ***, in connect
    _raise_connection_failure(self.address, error)
  File ***, line ***, in _raise_connection_failure
    raise AutoReconnect(msg) from error
pymongo.errors.AutoReconnect: localhost:27017: [Errno 8] nodename nor servname provided, or not known
```

This is one error related to the MongoDB while the MongoDB is running in the backend for a long time. The suggested solution is to restart the MongoDB. For more details such as how to restart the MongoDB in a specific environment, you can check the information [here](https://www.mongodb.com/docs/manual/installation/).

## Contributing
We are welcoming contributions from our collaborators. We strongly encourage contributors to work in their own forks of this project. You can edit the code on your own branch after you create your own forks. When you are ready to contribute, you can click the `Compare & pull` request button next to your own branch under your repository.

This project highly recommand committers to sign their code using the [Developer Certificate of Origin (DCO)](https://developercertificate.org/) for each git commit but this is not required.