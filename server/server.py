import collections
from flask import Flask, request, jsonify
from bson import json_util
import json


from calculate.database.db_util import * 

app = Flask(__name__)

## API Version 2.0

@app.route("/semi-global-local-structure-data", methods=["GET"])
def get_semi_global_local_structure_data():
    id = request.args.get("id")
    data = getDocument(SEMI_GLOBAL_LOCAL_STRUCTURE, {"caseId": id})
    print(data)
    return jsonify(data)


@app.route("/model-metadata", methods=["GET"])
def get_model_metadata():
    id = request.args.get("id")
    data = getDocument(MODEL_META_DATA, {"caseId": id})
    return jsonify(data)


@app.route("/loss-landscape-data", methods=["GET"])
def get_loss_landscape_data():
    id = request.args.get("id")
    modelIdModeId = request.args.get("modelIdModeId")
    modelId, modeId = modelIdModeId.split("-")
    data = getDocument(LOSS_LANDSCAPE, {"caseId": id, "modelId": modelId, "modeId": modeId})
    return jsonify(data)


@app.route("/persistence-barcode-data", methods=["GET"])
def get_persistence_barcode_data():
    id = request.args.get("id")
    modelIdModeId = request.args.get("modelIdModeId")
    modelId, modeId = modelIdModeId.split("-")
    data = getDocument(PERSISTENCE_BARCODE, {"caseId": id, "modelId": modelId, "modeId": modeId})
    return jsonify(data)


@app.route("/merge-tree-data", methods=["GET"])
def get_merge_tree_data():
    id = request.args.get("id")
    modelIdModeId = request.args.get("modelIdModeId")
    modelId, modeId = modelIdModeId.split("-")
    data = getDocument(MERGE_TREE, {"caseId": id, "modelId": modelId, "modeId": modeId})
    return jsonify(data)


@app.route("/layer-similarity-data", methods=["GET"])
def get_layer_similarity_data():
    id = request.args.get("id")
    modelId0 = request.args.get("modelId0")
    modeId0 = request.args.get("modeId0")
    modelId1 = request.args.get("modelId1")
    modeId1 = request.args.get("modeId1")

    modePairId = modelId0 + "_" + modeId0 + "_" + modelId1 + "_" + modeId1
    modePairIdAlt = modelId1 + "_" + modeId1 + "_" + modelId0 + "_" + modeId0

    query = {
        "$or": [
        {
            "caseId": id,
            "modePairId": modePairId,
        },
        {
            "caseId": id,
            "modePairId": modePairIdAlt,
        }
    ]
    }


    data = getDocument(LAYER_SIMILARITY, query)
    grid = data["grid"]
    flattenedGrid = []
    # there is a bug in the database for resnet, before fixing the data, we need to do this
    offset = 0
    if data["caseId"]== "resnet20":
        offset = -1
    for i in range(len(grid) + offset):
        for j in range(len(grid[i]) + offset):
            flattenedGrid.append(
                {
                    "xId": j,
                    "yId": i,
                    "value": grid[i][j]
                }
            )

    res = {
        "caseId": data["caseId"],
        "modePairId": data["modePairId"],
        "modelY": modelId0,
        "modelX": modelId1,
        "checkPointY": modeId0,
        "checkPointX": modeId1,
        "grid": flattenedGrid,
        "yLabels": data["xLabels"],
        "xLabels": data["yLabels"],
        "upperBound": data["upperBound"],
        "lowerBound": data["lowerBound"]
    }
    return jsonify(res)


@app.route("/confusion-matrix-bar-data", methods=["GET"])
def get_confusion_matrix_bar_data():
    id = request.args.get("id")
    modelId0 = request.args.get("modelId0")
    modeId0 = request.args.get("modeId0")
    modelId1 = request.args.get("modelId1")
    modeId1 = request.args.get("modeId1")
    modePairId = modelId0 + "_" + modeId0 + "_" + modelId1 + "_" + modeId1
    modePairIdAlt = modelId1 + "_" + modeId1 + "_" + modelId0 + "_" + modeId0

    query = {
        "$or": [
        {
            "caseId": id,
            "modePairId": modePairId,
        },
        {
            "caseId": id,
            "modePairId": modePairIdAlt,
        }
    ]
    }
    data = getDocument(CONFUSION_MATRIX, query)
    return jsonify(data)


@app.route("/regression-difference-data", methods=["GET"])
def get_regression_difference_data():
    id = request.args.get("id")
    modelId0 = request.args.get("modelId0")
    modeId0 = request.args.get("modeId0")
    modelId1 = request.args.get("modelId1")
    modeId1 = request.args.get("modeId1")
    modePairId = modelId0 + "_" + modeId0 + "_" + modelId1 + "_" + modeId1
    modePairIdAlt = modelId1 + "_" + modeId1 + "_" + modelId0 + "_" + modeId0

    query = {
        "$or": [
        {
            "caseId": id,
            "modePairId": modePairId,
        },
        {
            "caseId": id,
            "modePairId": modePairIdAlt,
        }
    ]
    }
    data = getDocument(REGRESSION_DIFFERENCE, query)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
