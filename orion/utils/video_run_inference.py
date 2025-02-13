import yaml
import torch

import wandb
from orion.utils.video_training_and_eval import perform_inference
from orion.utils.plot import initialize_classification_metrics, initialize_regression_metrics


def run_inference_and_log_to_wandb(
    config_path,
    wandb_id,
    model_path,
    resume,
    output_dir,
    split="test",
):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    config["model_path"] = config.get("model_path", model_path)
    config["wandb_id"] = wandb_id
    config["resume"] = resume
    config["debug"] = False
    config["output_dir"] = config.get("output_dir", output_dir)

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
        log_wandb=True,
    )

    return df_predictions_inference


def run_inference_and_no_logging(
    config_path, model_path=None, data_path=None, output_dir=None, split="inference"
):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Update config with provided arguments first
    config["data_filename"] = config.get("data_filename", data_path)
    config["model_path"] = config.get("model_path", model_path)
    config["output_dir"] = config.get("output_dir", output_dir)

    # Check for missing fields
    required_fields = ["model_path", "data_filename", "output_dir"]
    missing_fields = [
        field
        for field in required_fields
        if config.get(field) is None and locals().get(field) is None
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {', '.join(missing_fields)}. "
            "Please provide them either in the config file or as function parameters:\n"
            "- model_path: path to the model file\n"
            "- data_filename: path to the data file (provided as data_path parameter)\n"
            "- output_dir: directory for output files"
        )

    config["debug"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {split} inference on device {device}")

    return perform_inference(config=config, split=split, log_wandb=False)


def run_inference(
    config_path,
    split="test",
    log_wandb=False,
    model_path=None,
    data_path=None,
    output_dir=None,
    wandb_id=None,
    resume=False,
):
    if log_wandb:
        return run_inference_and_log_to_wandb(
            config_path, wandb_id, model_path, resume, output_dir, split
        )
    else:
        return run_inference_and_no_logging(config_path, model_path, data_path, output_dir, split)
