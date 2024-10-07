import sys
import os
from tqdm import tqdm
import argparse
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from script_util.update_db import *

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the case study for PINN.")
    parser.add_argument(
        "--compute", type=str, metavar="S", help="specify computation item", required=True
    )
    return parser.parse_args()

def load_model_info(model_name: str) -> Dict:
    path = os.path.join(parent_dir, "trained_models", model_name, "model_info.json")
    with open(path, "r") as f:
        return json.load(f)

def get_modes_list(model_list: List[str], case_id: str) -> List[Dict[str, str]]:
    modes_list = []
    for model_name in model_list:
        model_info = load_model_info(model_name)
        file_names = os.listdir(os.path.join(parent_dir, "trained_models", model_name))
        for file_name in file_names:
            if file_name == "model_info.json" or not file_name.startswith(model_name):
                continue
            mode_id = file_name.split("_")[-1].split(".")[0]
            modes_list.append({
                "caseId": case_id,
                "modelId": model_name,
                "modeId": mode_id,
            })
    return modes_list

def process_modes(modes_list: List[Dict[str, str]], compute: str) -> None:
    with tqdm(total=len(modes_list), desc="Progress", unit="iter", ncols=130) as pbar:
        for mode in modes_list:
            case_id = mode["caseId"]
            model_id = mode["modelId"]
            mode_id = mode["modeId"]

            if compute == "all" or compute == "performance":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: performance")
                update_mode_performance(case_id, model_id, mode_id)

            if compute == "all" or compute == "hessian":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: hessian")
                update_mode_hessian(case_id, model_id, mode_id)

            if compute == "all" or compute == "losslandscape":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: loss landscape")
                update_mode_losslandscape(case_id, model_id, mode_id)

            if compute == "all" or compute == "merge_tree":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: merge tree")
                update_mode_merge_tree(case_id, model_id, mode_id)

            if compute == "all" or compute == "persistence":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: persistence")
                update_mode_persistence_barcode(case_id, model_id, mode_id)

            if compute == "all" or compute == "layer_similarity":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: layer similarity")
                add_mode_layer_similarity(case_id, model_id, mode_id)

            if compute == "all" or compute == "mode_connectivity":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: mode connectivity")
                update_mode_connectivity(case_id, model_id, mode_id)

            if compute == "all" or compute == "confusion_matrix":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: confusion matrix")
                update_mode_confusion_matrix(case_id, model_id, mode_id)

            if compute == "all" or compute == "cka_similarity":
                pbar.set_postfix(Status=f"{model_id}-{mode_id}: cka similarity")
                update_mode_cka_similarity(case_id, model_id, mode_id)

            pbar.set_postfix(Status=f"{model_id}-{mode_id}: done")
            pbar.update(1)

def update_meta_data(meta_data_list: List[Dict], case_id: str) -> None:
    for meta_data in meta_data_list:
        model_id = meta_data["modelId"]
        update_model_meta_data(case_id, model_id, meta_data)

def main() -> None:
    args = parse_arguments()
    case_id = "pinn"
    model_list = ["PINN_convection_beta_1.0", "PINN_convection_beta_50.0"]

    meta_data_list = [load_model_info(model_name) for model_name in model_list]
    modes_list = get_modes_list(model_list, case_id)

    process_modes(modes_list, args.compute)
    update_meta_data(meta_data_list, case_id)

if __name__ == "__main__":
    main()
