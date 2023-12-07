import collections
from flask import Flask, escape, request, json, jsonify
import networkx as nx
from bson import json_util
import calculate.database.functions
import json

app = Flask(__name__)
# client = "mongodb://localhost:27017/"
client = 'mongodb'
database_name = "mydatabase"


# test route
@app.route("/")
def hello():
    """Just a test api

    Returns:
        String -- returned html
    """
    name = request.args.get("name", "World")
    return f"Hello, {escape(name)}!"


@app.route("/getHessianContour", methods=["POST"])
def get_hessian_contour_data():
    request_raw = request.get_json()
    dataset = request_raw["datasetName"]
    training = request_raw["training"]
    testing = request_raw["testing"]
    contour = request_raw["contourName"]
    modelType = request_raw["modelType"]
    if contour == "hessian":
        collection_name = "hessian_contour"
        query = {
            "dataset": dataset,
            "model_type": modelType,
            "testing": testing,
            "training": training,
        }
        result = calculate.database.functions.queryHessianContour(
            client, database_name, collection_name, query
        )
        # print(result)
        return jsonify(result)


@app.route("/getHessianContourModelInfo", methods=["POST"])
def get_hessian_contour_model_info():
    collection_name = "hessian_contour_model_information"
    request_raw = request.get_json()
    dataset = request_raw["datasetName"]
    modelType = request_raw["modelType"]
    training = request_raw["training"]
    testing = request_raw["testing"]
    step1 = request_raw["step1"]
    step2 = request_raw["step2"]
    query = {
        "dataset": dataset,
        "model_type": modelType,
        "testing": testing,
        "training": training,
        "step1": step1,
        "step2": step2,
    }
    result = calculate.database.functions.queryHessianContourModelInfo(
        client, database_name, collection_name, query
    )
    print(result)
    return jsonify(result)


@app.route("/getMetricViewInfo", methods=["POST"])
def get_hessian_contour_metricview_info():
    collection_name = "hessian_contour_model_information"
    request_raw = request.get_json()
    dataset = request_raw["datasetName"]
    modelType = request_raw["modelType"]
    training = request_raw["training"]
    testing = request_raw["testing"]
    step1 = request_raw["step1"]
    step2 = request_raw["step2"]
    query = {
        "dataset": dataset,
        "model_type": modelType,
        "testing": testing,
        "training": training,
        "step1": step1,
        "step2": step2,
    }
    result = calculate.database.functions.queryHessianContourMetricView(
        client, database_name, collection_name, query
    )
    # print(result)
    return jsonify(result)


@app.route("/getConfusionMatrix", methods=["POST"])
def get_hessian_contour_confusion_matrix():
    collection_name = "hessian_contour_model_information"
    request_raw = request.get_json()
    dataset = request_raw["datasetName"]
    modelType = request_raw["modelType"]
    training = request_raw["training"]
    testing = request_raw["testing"]
    step1 = request_raw["step1"]
    step2 = request_raw["step2"]
    query = {
        "dataset": dataset,
        "model_type": modelType,
        "testing": testing,
        "training": training,
        "step1": step1,
        "step2": step2,
    }
    result = calculate.database.functions.queryHessianContourConfusionMatrix(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getMergeTree", methods=["POST"])
def get_merge_tree():
    collection_name = "ttk_merge_tree"
    request_raw = request.get_json()
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    dataset = request_raw["dataset"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.queryMergeTree(
        client, database_name, collection_name, query
    )
    # print(result)
    return jsonify(result)


@app.route("/getMergeTreeTraining", methods=["POST"])
def get_merge_tree_training():
    collection_name = "ttk_merge_tree_training"
    request_raw = request.get_json()
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    dataset = request_raw["dataset"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.queryMergeTree(
        client, database_name, collection_name, query
    )
    # print(result)
    return jsonify(result)


@app.route("/getPersistantDiagram", methods=["POST"])
def get_persistant_diagram():
    collection_name = "ttk_persistant_diagram"
    request_raw = request.get_json()
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    dataset = request_raw["dataset"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.queryPersistantDiagram(
        client, database_name, collection_name, query
    )
    # print(result)
    return jsonify(result)


@app.route("/getPersistantDiagramTraining", methods=["POST"])
def get_persistant_diagram_training():
    collection_name = "ttk_persistant_diagram_training"
    request_raw = request.get_json()
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    dataset = request_raw["dataset"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.queryPersistantDiagram(
        client, database_name, collection_name, query
    )
    # print(result)
    return jsonify(result)


@app.route("/getLossContour", methods=["POST"])
def get_loss_contour():
    collection_name = "loss_landscapes_contour"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.queryLossContour(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getTrainLossContour", methods=["POST"])
def get_train_loss_contour():
    collection_name = "training_loss_landscape"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.queryTrainLossContour(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getLossModelInfo", methods=["POST"])
def get_loss_model_info():
    collection_name = "model_analysis_information"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.queryLossModelInfo(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getPredictionDistribution", methods=["POST"])
def get_prediction_distribution():
    collection_name = "prediction_distribution"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model1 = request_raw["model1"]
    model2 = request_raw["model2"]
    threshold = request_raw["threshold"]
    query = {"dataset": dataset, "model_type": modelType, "model1": model1, "model2": model2, "threshold": threshold}
    result = calculate.database.functions.queryPredictionDistribution(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getPredictionDistributionLabels", methods=["POST"])
def get_prediction_distribution_labels():
    collection_name = "prediction_distribution_labels"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    query = {"dataset": dataset}
    result = calculate.database.functions.queryPredictionDistributionLabels(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getLossGlobal", methods=["POST"])
def get_loss_global():
    collection_name = "loss_landscapes_global"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    query = {"dataset": dataset, "model_type": modelType}
    result = calculate.database.functions.queryLossGlobal(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getHessianLossContour", methods=["POST"])
def get_hessian_loss_contour():
    collection_name = "hessian_loss_landscape"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.queryHessianLossContour(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/get3DLossContour", methods=["POST"])
def get_3d_loss_contour():
    collection_name = "loss_landscapes_contour_3d"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.query3DLossContour(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/get3DTrainLossContour", methods=["POST"])
def get_3d_loss_contour_train():
    collection_name = "loss_landscapes_contour_3d_training"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model = request_raw["model"]
    query = {"dataset": dataset, "model_type": modelType, "model": model}
    result = calculate.database.functions.query3DTrainLossContour(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getModelWeightSimilarity", methods=["POST"])
def get_model_weight_similarity():
    collection_name = "model_weight_similarity"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    modelOne = request_raw["modelOne"]
    modelTwo = request_raw["modelTwo"]
    query = {
        "dataset": dataset,
        "model_type": modelType,
        "modelOne": modelOne,
        "modelTwo": modelTwo,
    }
    result = calculate.database.functions.queryModelWeightSimilarity(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getLayerMDSSimilarity", methods=["POST"])
def get_layer_mds_similarity():
    collection_name = "layer_euclidean_distance_similarity"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    query = {"dataset": dataset, "model_type": modelType}
    result = calculate.database.functions.queryLayerMDSSimilarity(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getTorchvisionLayerMDSSimilarity", methods=["POST"])
def get_torchvision_layer_mds_similarity():
    collection_name = "layer_euclidean_distance_similarity"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    query = {"dataset": dataset, "model_type": modelType}
    result = calculate.database.functions.queryTorchvisionLayerMDSSimilarity(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getTorchvisionLayerCKAMDSSimilarity", methods=["POST"])
def get_torchvision_layer_cka_mds_similarity():
    collection_name = "layer_cka_similarity"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    query = {"dataset": dataset, "model_type": modelType}
    result = calculate.database.functions.queryTorchvisionLayerCKAMDSSimilarity(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getLayerTorchCKASimilarity", methods=["POST"])
def get_layer_torch_cka_similarity():
    collection_name = "layer_torch_cka_similarity"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model1 = request_raw["model1"]
    model2 = request_raw["model2"]
    query = {"dataset": dataset, "model_type": modelType, "model1": model1, "model2": model2}
    result = calculate.database.functions.queryLayerTorchCKASimilarity(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getModelMDSSimilarity", methods=["POST"])
def get_model_mds_similarity():
    collection_name = "model_euclidean_distance_similarity"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    query = {"dataset": dataset, "model_type": modelType}
    result = calculate.database.functions.queryModelMDSSimilarity(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getLossDetailedModelSimilarity", methods=["POST"])
def get_loss_detailed_model_similarity():
    collection_name = "loss_landscapes_detailed_similarity"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    model1 = request_raw["model1"]
    model2 = request_raw["model2"]
    threshold = request_raw["threshold"]
    query = {
        "dataset": dataset,
        "model_type": modelType,
        "model1": model1,
        "model2": model2,
        "threshold": threshold,
    }
    result = calculate.database.functions.queryLossDetailedModelSimilarity(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getModelGlobalStructure", methods=["POST"])
def get_model_global_structure():
    collection_name = "global_structure"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    query = {"dataset": dataset, "model_type": modelType}
    result = calculate.database.functions.queryModelGlobalStructure(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)


@app.route("/getModelCKASimilarity", methods=["POST"])
def get_model_cka_similarity():
    collection_name = "cka_model_similarity"
    request_raw = request.get_json()
    dataset = request_raw["dataset"]
    modelType = request_raw["modelType"]
    query = {"dataset": dataset, "model_type": modelType}
    result = calculate.database.functions.queryModelCKASimilarity(
        client, database_name, collection_name, query
    )
    #print(result)
    return jsonify(result)

## API Version 2.0

@app.route("/semi-global-local-structure-data", methods=["GET"])
def get_semi_global_local_structure_data():
    id = request.args.get("id")
    with open("./temp_data/semi-global-local-structure-data-" + id + ".json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/model-metadata", methods=["GET"])
def get_model_metadata():
    id = request.args.get("id")
    with open("./temp_data/model-metadata-" + id + ".json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/loss-landscape-data", methods=["GET"])
def get_loss_landscape_data():
    id = request.args.get("id")
    modeId = request.args.get("modeId")
    with open("./temp_data/loss-landscape-data-" + id + "-" + modeId + ".json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/persistence-barcode-data", methods=["GET"])
def get_persistence_barcode_data():
    id = request.args.get("id")
    modeId = request.args.get("modeId")
    with open("./temp_data/persistence-barcode-data-" + id + "-" + modeId + ".json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/merge-tree-data", methods=["GET"])
def get_merge_tree_data():
    id = request.args.get("id")
    modeId = request.args.get("modeId")
    with open("./temp_data/merge-tree-data-" + id + "-" + modeId + ".json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/layer-similarity-data", methods=["GET"])
def get_layer_similarity_data():
    id = request.args.get("id")
    modeId0 = request.args.get("modeId0")
    modeId1 = request.args.get("modeId1")

    with open("./temp_data/layer-similarity-data-" + id + "-" + modeId0 + "-" + modeId1 +".json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/confusion-matrix-bar-data", methods=["GET"])
def get_confusion_matrix_bar_data():
    id = request.args.get("id")
    modeId0 = request.args.get("modeId0")
    modeId1 = request.args.get("modeId1")
    with open("./temp_data/confusion-matrix-bar-data-" + id + "-" + modeId0 + "-" + modeId1 +".json") as f:
        data = json.load(f)
    return jsonify(data)



if __name__ == "__main__":
    app.run(debug=True)
