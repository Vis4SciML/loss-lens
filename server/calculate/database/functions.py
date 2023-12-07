from pickle import TRUE
import pymongo

USER_NAME = "jiaqingchen"
PASS_WORD = "BdGtuIC&oT77K$Fi"
AUTH_SOURCE = "losslensdb"

def getClient(client):
    if client == "mongodb07.nersc.gov":
        myclient = pymongo.MongoClient(host=client, username=USER_NAME, password=PASS_WORD, authSource=AUTH_SOURCE)
    elif client == "mongodb":
        myclient = pymongo.MongoClient(host=client, username="user", password="pass", authSource="admin")
    else:
        myclient = pymongo.MongoClient(client)
    
    return myclient

def create(client, database_name, collection_name):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
        
def empty(client, database_name, collection_name):
    myclient = getClient(client)
    mydb = myclient[database_name]
    collist = mydb.list_collection_names()
    if collection_name in collist:
        return False
    else:
        return True

def insert(client, database_name, collection_name, record):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    x = mycol.insert_one(record)
    return x.inserted_id

def findOne(client, database_name, collection_name):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    x = mycol.find_one()
    return x

def findAll(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query):
        queryResultList.append(x)
    return queryResultList

def queryHessianContour(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0, 'dataset': 1, 'model_type': 1, 'testing': 1, 'training': 1, 'steps': 1, 'start': 1, 'end': 1, 'top_eigenvalues': 1, 'result': 1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]
    
def queryHessianContourModelInfo(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0,'dataset':1, 'model_type': 1, 'testing':1,'training':1,'step1':1,'step2':1,'accuracy':1, 'recall':1, 'precision':1, 'f1':1, 'confusionMatrix':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryHessianContourMetricView(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0,'dataset':1, 'model_type': 1, 'testing':1,'training':1,'step1':1,'step2':1,'accuracy':1, 'recall':1, 'precision':1, 'f1':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryHessianContourConfusionMatrix(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0,'dataset':1, 'model_type': 1, 'testing':1,'training':1,'step1':1,'step2':1,'confusionMatrix':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryHessianContourWeightBias(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0,'dataset':1, 'model_type': 1, 'testing':1,'training':1,'step1':1,'step2':1,'model_layer1_weight':1,'model_layer1_bias':1,'model_layer2_weight':1,'model_layer2_bias':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryMergeTree(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0, 'dataset': 1, 'model_type': 1, 'model':1, 'start':1, 'end':1, 'x':1, 'y':1, 'z':1, 'nodeID':1,'branchID':1}):
        queryResultList.append(x)
    return queryResultList

def queryPersistantDiagram(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0, 'dataset': 1, 'model_type': 1, 'model':1, 'coordinates_0':1, 'coordinates_1':1, 'coordinates_2':1, 'nodeID':1}):
        queryResultList.append(x)
    return queryResultList

def queryLossContour(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0, "dataset":1, 'model_type': 1, "model":1, "steps":1, "loss_data_fin":1, "max_loss_value_x":1, "max_loss_value_y":1, "min_loss_value_x":1, "min_loss_value_y":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryTrainLossContour(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0, "dataset":1, 'model_type': 1, "model":1, "steps":1, "loss_data_fin":1, "max_loss_value_x":1, "max_loss_value_y":1, "min_loss_value_x":1, "min_loss_value_y":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryLossModelInfo(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0, 'dataset':1, 'model_type':1, 'model':1,'accuracy':1,'recall':1,'precision':1,'f1':1,'confusionMatrix':1, 'top_eigenvalues':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]
    
def queryLossGlobal(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0, "dataset":1, 'model_type': 1, "steps":1, "loss_data_fin":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def query3DLossContour(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0, "dataset":1, 'model_type': 1, "model":1, "steps":1, "loss_data_fin_3d":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def query3DTrainLossContour(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0, "dataset":1, 'model_type': 1, "model":1, "steps":1, "loss_data_fin_3d":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryHessianLossContour(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0, "dataset":1, 'model_type': 1, "model":1, "steps":1, "loss_data_fin":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryPredictionDistribution(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0, "dataset":1, 'model_type':1, "model1":1, "model2":1, "correct_correct":1, "correct_wrong":1, "wrong_correct":1, "wrong_wrong":1, "threshold":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryPredictionDistributionLabels(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0, "dataset":1, "labels":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]
    
def queryModelWeightSimilarity(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0,"dataset":1,'model_type': 1,"modelOne":1,"modelTwo":1,"similarityOneTwoFirst":1,"similarityOneTwoSecond":1,"similarityOneAllFirst":1,"similarityOneAllSecond":1,"similarityTwoAllFirst":1,"similarityTwoAllSecond":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryLayerMDSSimilarity(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0,"dataset":1,'model_type':1,"layerOneMDS":1,"layerTwoMDS":1,'layerOneMatrix':1,'layerTwoMatrix':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryTorchvisionLayerMDSSimilarity(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0,"dataset":1,'model_type':1, "index":1, "modelLayerMDS":1,"modelLayerMatrix":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList

def queryTorchvisionLayerCKAMDSSimilarity(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0,'dataset':1, 'model_type':1, 'index':1, 'layer1_linearCKA_MDS':1, 'layer1_kernelCKA_MDS':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList

def queryLayerTorchCKASimilarity(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{'_id': 0,'dataset':1, 'model_type':1, 'model1':1, 'model2':1, 'torchCKA_similarity':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList

def queryLossDetailedModelSimilarity(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0,"dataset":1,'model_type':1,"model1":1,"model2":1,"modelMDS":1,"threshold":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]
    
def queryModelMDSSimilarity(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0,"dataset":1,'model_type':1,"modelMDS":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryModelGlobalStructure(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0,"dataset":1,'model_type':1,"global_structure":1, 'linearCKA_global_structure':1, 'kernelCKA_global_structure':1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def queryModelCKASimilarity(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    queryResultList = []
    for x in mycol.find(query,{"_id": 0,"dataset":1,'model_type':1,"linearCKA_MDS":1,"kernelCKA_MDS":1}):
        queryResultList.append(x)
    if(len(queryResultList) != 1):
        print("There is more than one result in this query, return the first one.")
    return queryResultList[0]

def sort(client, database_name, collection_name, key):
    if client == "mongodb":
        myclient = pymongo.MongoClient(host=client, username="user", password="pass", authSource="admin")
        mydb = myclient[database_name]
        mycol = mydb[collection_name]
    else:
        myclient = getClient(client)
        mydb = myclient[database_name]
        mycol = mydb[collection_name]
    mydoc = mycol.find().sort(key)
    print("Print the results after sorting.")
    for x in mydoc:
        print(x)

def sortDescending(client, database_name, collection_name, key):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    mydoc = mycol.find().sort(key, -1)
    print("Print the results after sorting.")
    for x in mydoc:
        print(x)

def deleteOne(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    mycol.delete_one(query)
    print("Successfully delete one document.")
    return True

def deleteMany(client, database_name, collection_name, query):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    x = mycol.delete_many(query)
    print(x.deleted_count, " documents have been successfully deleted.")
    return True

def drop(client, database_name, collection_name):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    mycol.drop()
    print("Successfully drop one collection.")
    return True

def limit(client, database_name, collection_name, number):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    myresult = mycol.find().limit(number)
    queryResultList = []
    for x in myresult:
        queryResultList.append(x)
    return queryResultList

def updateOne(client, database_name, collection_name, query, new_values):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    mycol.update_one(query, new_values)
    print("Print the results after updating one document.")
    for x in mycol.find():
        print(x)

def updateMany(client, database_name, collection_name, query, new_values):
    myclient = getClient(client)
    mydb = myclient[database_name]
    mycol = mydb[collection_name]
    x = mycol.update_many(query, new_values)
    print(x.modified_count, "documents have been successfully updated.")
