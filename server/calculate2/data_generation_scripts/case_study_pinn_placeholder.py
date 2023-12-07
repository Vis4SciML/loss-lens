import sys
import os
from tqdm import tqdm
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from script_util.update_db_placeholder import *

parser = argparse.ArgumentParser(description="Run the case study for PINN.")
parser.add_argument(
    "--compute", type=str, metavar="S", help="specify computation item", required=True
)

args = parser.parse_args()
case_id = "pinn"

model_list = ["pinn_convection_beta1", "pinn_convection_beta50"]
meta_data_list = []
modes_list = []

for model_name in model_list:
    # print("model_name: ", model_name)
    path = parent_dir + "/trained_models/" + model_name + "/model_info.json"
    with open(path, "r") as f:
        model_info = json.load(f)
    meta_data_list.append(model_info)

    file_names = os.listdir(parent_dir + "/trained_models/" + model_name)
    print(file_names)
    for file_name in file_names:
        if file_name == "model_info.json" or not file_name.startswith(model_name):
            continue
        mode_id = file_name.split(".")[0].split("_")[-1]
        mode = {
            "caseId": case_id,
            "modelId": model_name,
            "modeId": mode_id,
        }
        modes_list.append(mode)

print("meta_data_list: ", meta_data_list)
print("modes_list: ", modes_list)


with tqdm(total=len(modes_list), desc="Progress", unit="iter", ncols=100) as pbar:
    for i in range(len(modes_list)):
        mode = modes_list[i]
        case_id = mode["caseId"]
        model_id = mode["modelId"]
        mode_id = mode["modeId"]
        if args.compute == "all" or args.compute == "performance":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: performance")
            update_mode_performance(case_id, model_id, mode_id)

        if args.compute == "all" or args.compute == "hessian":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: hessian")
            update_mode_hessian(case_id, model_id, mode_id)

        if args.compute == "all" or args.compute == "losslandscape":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: loss landscape")
            update_mode_losslandscape(case_id, model_id, mode_id)

        if args.compute == "all" or args.compute == "merge_tree":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: merge tree")
            update_mode_merge_tree(case_id, model_id, mode_id)

        if args.compute == "all" or args.compute == "persistence":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: persistence")
            update_mode_persistence_barcode(case_id, model_id, mode_id)

        if args.compute == "all" or args.compute == "layer_similarity":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: layer similarity")
            update_mode_layer_similarity(case_id, model_id, mode_id)

        if args.compute == "all" or args.compute == "mode_connectivity":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: mode connectivity")
            update_mode_connectivity(case_id, model_id, mode_id)

        if args.compute == "all" or args.compute == "confusion_matrix":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: confusion matrix")
            update_mode_confusion_matrix(case_id, model_id, mode_id)

        if args.compute == "all" or args.compute == "cka_similarity":
            pbar.set_postfix(Status=f"{model_id}-{mode_id}: cka similarity")
            update_mode_cka_similarity(case_id, model_id, mode_id) 


        pbar.set_postfix(Status=f"{model_id}-{mode_id}: done")
        pbar.update(1)


for meta_data in meta_data_list:
    model_id = meta_data["modelId"]
    update_model_meta_data(case_id, model_id, meta_data)
