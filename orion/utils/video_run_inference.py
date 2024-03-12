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
    config["output_dir"] = checkpoints_folder

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

    save_predictions_to_csv(df_predictions_inference, config, split)

    return df_predictions_inference


def run_inference_and_no_logging(
    config_path, model_path=None, data_path=None, output_dir=None, split="inference"
):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    required_fields = ["model_path", "data_filename", "output_dir"]

    missing_fields = [
        field
        for field in required_fields
        if config.get(field) is None and locals().get(field) is None
    ]

    if missing_fields:
        raise ValueError(f"Missing required config fields: {', '.join(missing_fields)}")

    config["data_filename"] = config.get("data_filename", data_path)
    config["model_path"] = config.get("model_path", model_path)
    config["output_dir"] = config.get("output_dir", output_dir)
    config["debug"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_predictions_inference = perform_inference(config=config, split=split)
    return df_predictions_inference
