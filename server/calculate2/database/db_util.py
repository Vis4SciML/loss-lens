from pickle import TRUE
import pymongo
import json
import os
from typing import Dict

SEMI_GLOBAL_LOCAL_STRUCTURE = "semi-global-local-structure"
LOSS_LANDSCAPE = "loss-landscape"
MERGE_TREE = "merge-tree"
PERSISTENCE_BARCODE = "persistence-barcode"
LAYER_SIMILARITY = "layer-similarity"
CONFUSION_MATRIX = "confusion-matrix"
CKA_SIMILARITY = "cka-similarity"
MODEL_META_DATA = "model-metadata"
REGRESSION_DIFFERENCE = "regression-difference"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
with open(os.path.join(current_dir, "credential.json"), "r") as f:
    credential = json.load(f)
with open(os.path.join(parent_dir, "db_client_type.json"), "r") as f:
    db_client_type = json.load(f)
clientInfo = credential[db_client_type["clientType"]]


def getClient():
    client = pymongo.MongoClient(
        host=clientInfo["host"],
        username=clientInfo["username"],
        password=clientInfo["password"],
        authSource=clientInfo["authSource"],
    )
    return client


def dbExists():
    client = getClient()
    dblist = client.list_database_names()
    if clientInfo["database"] in dblist:
        return True
    else:
        return False


def createDB():
    client = getClient()
    db = client[clientInfo["database"]]


def collectionExists(collection_name: str):
    client = getClient()
    db = client[clientInfo["database"]]
    collist = db.list_collection_names()
    if collection_name in collist:
        return True
    else:
        return False


def createCollection(collection_name: str):
    client = getClient()
    db = client[clientInfo["database"]]
    collection = db[collection_name]


def addOrUpdateDocument(
    collection_name: str, query: Dict[str, any], record: Dict[str, any]
):
    client = getClient()
    db = client[clientInfo["database"]]
    collection = db[collection_name]
    query_res = collection.find_one(query)
    if query_res is None:
        for key in record.keys():
            query[key] = record[key]
        collection.insert_one(query)
    else:
        record = {"$set": record}
        collection.update_one(query, record)


def getDocuments(collection_name: str, query: Dict[str, any]):
    client = getClient()
    db = client[clientInfo["database"]]
    collection = db[collection_name]
    exclusion = {"_id": 0}
    query_res = collection.find(query, exclusion)
    return query_res


def getDocument(collection_name: str, query: Dict[str, any]):
    client = getClient()
    db = client[clientInfo["database"]]
    collection = db[collection_name]
    exclusion = {"_id": 0}
    query_res = collection.find_one(query, exclusion)
    return query_res
