# Back-End Instructions

## Package Install Instructions

### Official Packages
Back-End server requires several official packages which we do not need to make any changes. We can directly use those packages. They are all included in the `requirements.txt` under the current folder. You can either install them in your own way or install all of those packages as follows:

```shell
pip install -r requirements.txt
```

### Modified Packages
Besides the official packages we mentioned above, back-end server also requires one modified package, [loss_landscapes](https://github.com/marcellodebernardi/loss-landscapes). We have made some changes to the package and it is required to install this modified package manually in the following way from current folder.

```shell
pip install calculate/operation/dist/loss_landscapes-3.0.7.tar.gz
```

This modified `loss_landscapes` package source code can be found under `calculate/operation/` folder. For more information, you can either refer the original [loss_landscapes](https://github.com/marcellodebernardi/loss-landscapes) or [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## MongoDB Set-Up Instructions
After we have the trained model and also the datasets ready, we need to calculate the loss values and also other meaningful results for visualization. In this project, we use the [MongoDB](https://www.mongodb.com/) to store the calculation results which can be requested by the Front-End for visualization at any time. There are many ways to install the MongoDB in different platform. You can find more details in the [MongoDB Installation](https://www.mongodb.com/docs/manual/installation/). Here, we provide one way to install the MongoDB in Linux Ubuntu as follows.

First, we need to import the public key used by the package management system:

```shell
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
```

Then, we need to create the /etc/apt/sources.list.d/mongodb-org-6.0.list file for Ubuntu. Here ia an example for Ubuntu 20.04 (Focal):

```shell
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
```

Next, we need to install the MongoDB packagee. You can install either the latest stable version of MongoDB or a specific version of MongoDB. You might need to update the `apt-get` at first and to install the latest stable version, run the following command.

```shell
sudo apt-get install -y mongodb-org
```

Although you can specify any available version of MongoDB, `apt-get` will upgrade the packages when a newer version becomes available. To prevent unintended upgrades, you can pin the package at the currently installed version:

```shell
echo "mongodb-org hold" | sudo dpkg --set-selections
echo "mongodb-org-database hold" | sudo dpkg --set-selections
echo "mongodb-org-server hold" | sudo dpkg --set-selections
echo "mongodb-mongosh hold" | sudo dpkg --set-selections
echo "mongodb-org-mongos hold" | sudo dpkg --set-selections
echo "mongodb-org-tools hold" | sudo dpkg --set-selections
```

At this point, you should have your MongoDB ready.

And we need to start the MongoDB. You can start the mongod process by issuing the following command:

```shell
sudo systemctl start mongod
```

You can also verify that MongoDB has started successfully by using the following command:

```shell
sudo systemctl status mongod
```

You can also stop the MongoDB by using the following commands:

```shell
sudo systemctl stop mongod
```

## Training model Instructions
The first step to visualize several models in this tool is to prepare several pre-trained models for comparison. We provide some examples under the [train](train/) folder. You can also make your own training code under the [train](train/) folder or you can directly put your own pre-trained models under the [model](model/) folder. For training models, you can refer the [Training model Instructions](train/README.md#training-model-instructions).

## Calculation for Visualization Instructions
To generate the data which can be used for the visualization in the Front-End, you need to do the calculation under the [calculate](calculate/) folder. You can refer the [Server Pipeline Instructions](calculate/README.md#server-pipeline-instructions). All the results will be saved in the MongoDB database. Once the server has started and received requests from the Front-End, `server.py` will return corresponding results.

## Start the Back-End Server
Once all the required loss visualization values are correctly calculated and stored in the MongoDV database, we need to start the Back-End server. The Back-End server can be started by the following command:

```shell
python3 server.py
```

Note that in order to successfully get all the results from the MongoDB database, before having communication with the Front-End requests, make sure that you have started the MongoDB and have stable connection with it.