# Adding the case study

We provide the pipeline making it easy to add case study based on your own model and data. The infrastructure of the pipeline is this:

```
server
    |
    |- calculate                      # preprocessing scripts
    |   |- data                       # datasets
    |   |   |- CIFAR10
    |   |   |- CIFAR10C
    |   |   |- MNIST
    |   |   |- ...
    |   |- data_generation_scripts
    |   |   |- case_study_mlp.py      # such scripts can pass parameters to allow
    |   |   |- case_study_vit.py      # specific computation
    |   |   |- ...
    |   |   |- run_case_studies.sh    # modify this to compute specific items
    |   |   |- run.sh                 # scripts to run on cluster computers
    |   |- database
    |   |   |- credential.json        # customized user and pass to connect to db
    |   |   |- db_util.py             # util files for operating data with db
    |   |- db_client_type.json        # specify db host to load config automatically
    |   |- loss_landscape             # code for generating loss cubes
    |   |- script_util                # util files to do all computation
    |   |   |- core_functions.py      # core functions for computing metrics of models
    |   |   |- torch_cka              # for computing cka similarity
    |   |- trained_models             # all trained models
    |   |   |- cifar10_augvit
    |   |   |- ...
    |   |- training_scripts           # scripts for training models and their architectures
    |- server.py                      # server scripts for hosting the apis
    |- legacy_files                   # legacy, do not refer to those

```

## Training the models

We put training scripts as well as the model class declaritions in `calculate/training_scripts`, and the trained models will be stored at `calculate/trained_models`.

## Running your own case studies

To run a case study, first check `credential.json` and `db_client_type.json`. For different mongodb hosts, we will have different credential to connected to the database. Thus, modify them based on your needs.

Runing scripts to generate case studies requires modifying two key files, `case_study_[name].py` and `core_functions.py`. The `case_study_[name].py` is for configuring what metric to compute and how to store them in the database. The `core_functions.py` is for configuring how the metrics are computed, such as CKA and Mode Connectivity. You might want to create new metrics here.

Note that, we want this process very simple by executing the `case_study_[name].py` to get all necessary data processing done, and store the result in the database, so that the case study can be directly used and visualized. Thus, the idea of this file is to specify all model names, and the scripts will find all models with names matched in the `calculate/trained_models`, load them and compute all the metrics.

## After the Case Study

Since all the computed data is either stored in the local machine or your own docker containers, remember to dump the database as the backup and overwrite the `database/losslensdb` in the root folder, so that next time you will have consistent result by restoring that backup.

To do so, run:

```
docker exec [docker-container-id] mongodump --username admin --password pass --authenticationDatabase=admin --db losslensdb --out=/docker-entrypoint-initdb.d/
```

This will automatically backup the current database and save files to `/database/`.

To look for the `[docker-container-id]`, run `docker ps` in the terminal and locate the mongodb container's ID.
