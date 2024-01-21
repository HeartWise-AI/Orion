import os

import torch
import yaml

import wandb
from orion.utils.plot import initialize_classification_metrics, initialize_regression_metrics
from orion.utils.video_training_and_eval import perform_inference


def run_inference_and_log_to_wandb(
    checkpoints_folder, model_file_name, wandb_id, resume, config_path, split="test"
):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    config["model_path"] = os.path.join(checkpoints_folder, model_file_name)
    config["wandb_id"] = wandb_id
    config["resume"] = resume
    config["debug"] = False
    config["output"] = checkpoints_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["task"] == "classification":
        num_classes = 12 if config["num_classes"] <= 1 else config["num_classes"]
        metrics = {
            "train": initialize_classification_metrics(num_classes, device),
            "val": initialize_classification_metrics(num_classes, device),
            "test": initialize_classification_metrics(num_classes, device),
        }
    else:
        metrics = {
            "train": initialize_regression_metrics(device),
            "val": initialize_regression_metrics(device),
            "test": initialize_regression_metrics(device),
        }

    best_metrics = {
        "val": {
            "best_loss": float("inf"),
            "best_auc": -float("inf"),
            "optimal_thresh": 0.5,
            "best_mae": float("inf"),
            "best_rmse": float("inf"),
        },
        "test": {
            "best_loss": float("inf"),
            "best_auc": -float("inf"),
            "optimal_thresh": 0.5,
            "best_mae": float("inf"),
            "best_rmse": float("inf"),
        },
    }

    wandb.init(
        entity=config["entity"],
        project=config["project"],
        config=config,
        name=config["project"],
        resume=config.get("resume", False),
        id=config.get("wandb_id", None),
    )

    df_predictions_inference = perform_inference(
        config=config,
        split=split,
        metrics=metrics,
        best_metrics=best_metrics,
    )

    return df_predictions_inference


def run_inference_and_no_logging(checkpoints_folder, data_path, model_file_name, config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    config["data_filename"] = data_path
    config["model_path"] = os.path.join(checkpoints_folder, model_file_name)
    config["debug"] = False
    config["output"] = checkpoints_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_predictions_inference = perform_inference(config=config, split="inference")

    return df_predictions_inference
