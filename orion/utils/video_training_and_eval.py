import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms

import wandb

from typing import List, Dict, Optional

# Add the parent directory of 'orion' to the Python path
dir2 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if dir2 not in sys.path:
    sys.path.append(dir2)

import tqdm

import orion
from orion.datasets import Video
from orion.models import (
    load_and_modify_model,
    movinet,
    stam,
    timesformer,
    vivit,
    x3d_legacy,
    x3d_multi,
)
from orion.models.videopairclassifier import VideoPairClassifier
from orion.utils import arg_parser, dist_eval_sampler, plot, video_training_and_eval
from orion.utils.plot import (
    bootstrap_metrics,
    bootstrap_multicalss_metrics,
    compute_classification_metrics,
    compute_multiclass_metrics,
    compute_optimal_threshold,
    initialize_classification_metrics,
    initialize_regression_metrics,
    log_binary_classification_metrics_to_wandb,
    log_multiclass_metrics_to_wandb,
    log_regression_metrics_to_wandb,
    metrics_from_moving_threshold,
    plot_moving_thresh_metrics,
    plot_multiclass_confusion,
    plot_multiclass_rocs,
    plot_preds_distribution,
    plot_regression_graphics_and_log_binarized_to_wandb,
    update_best_regression_metrics,
    update_classification_metrics,
)
from orion.utils.losses import LossRegistry

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

torch.set_float32_matmul_precision("high")


def execute_run(config_defaults=None, transforms=None, args=None, run=None):
    """
    Executes the training and evaluation process for a given configuration.

    Args:
        config_defaults (dict): Default configuration values.
        transforms (list): List of data transforms.
        args (argparse.Namespace): Command-line arguments.
        run (wandb.Run): WandB run object for logging.

    Returns:
        None
    """
    torch.cuda.empty_cache()
    # Check to see if local_rank is 0
    is_master = args.local_rank == 0
    print("is_master", is_master)

    config = setup_config(config_defaults, transforms, is_master, run=run)
    use_amp = config.get("use_amp", False)
    print("Using AMP", use_amp)
    task = config.get("task", "regression")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    do_log = run is not None  # Run is none if process rank is not 0

    # set the device
    total_devices = torch.cuda.device_count()
    if total_devices == 0:
        raise RuntimeError("No CUDA devices available. This script requires CUDA.")
    device = torch.device(args.local_rank % total_devices)
    print("Total devices", total_devices)
    print("Device ", device)

    # initialize PyTorch distributed using environment variables
    dist.init_process_group(backend="nccl", init_method="env://")
    is_main_process = dist.get_rank() == 0
    print("is main process", is_main_process)
    torch.cuda.set_device(device)

    # Use model_path from config, or default to config["output_dir"] if model_path is None
    model_path = config.get("model_path") or config.get("output_dir")
    if "binary_threshold" not in config:
        config["binary_threshold"] = 0.5
        print("Error: 'binary_threshold' not found in config. Setting it to default value 0.5.")

    # Extract the last element of the folder path
    if do_log:
        last_folder_name = os.path.basename(model_path)
        entity = run.entity
        project = run.project
        run_id = run.id

        # Accessing the run using the W&B API
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        # Assuming last_folder_name is a string containing the name of the last folder
        # Concatenate "_" with last_folder_name to form the new tag
        new_tag = "_" + last_folder_name

        # Ensure new_tag is a list before concatenating
        run.tags = run.tags + [new_tag]

        run.update()

    # Model building and training setup
    (
        model,
        optim_state,
        sched_state,
        epoch_resume,
        bestLoss,
        other_metrics,
        labels_map,
    ) = build_model(config, device, model_path=model_path)

    # watch gradients only for rank 0
    if is_master:
        wandb.watch(model)

    ### If PyTorch 2.0 is used, the following line is needed to load the model

    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, epoch_resume)

    # If optimizer and scheduler states were loaded, apply them
    if optim_state:
        print("Resuming with the previous optimizer state")
        optimizer.load_state_dict(optim_state)

    if sched_state:
        print("Resuming with the previous scheduler state", sched_state)
        scheduler.load_state_dict(sched_state)

    ## Data loader
    train_dataset = load_dataset("train", config, transforms, config["weighted_sampling"])
    val_dataset = load_dataset("val", config, transforms, config["weighted_sampling"])

    # Simplified DataLoader setup
    def create_dataloader(dataset, sampler_type, batch_size, num_workers, pin_memory, drop_last):
        sampler = sampler_type(dataset)
        return (
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
                pin_memory=pin_memory,
                drop_last=drop_last,
            ),
            sampler,
        )

    train_sampler_type = (
        WeightedRandomSampler
        if config["weighted_sampling"]
        else torch.utils.data.distributed.DistributedSampler
    )

    train_loader, train_sampler = create_dataloader(
        train_dataset,
        train_sampler_type,
        config["batch_size"],
        config["num_workers"],
        device.type == "cuda",
        True,
    )
    val_sampler_type = torch.utils.data.distributed.DistributedSampler
    val_loader, val_sampler = create_dataloader(
        val_dataset,
        val_sampler_type,
        config["batch_size"],
        config["num_workers"],
        device.type == "cuda",
        True,
    )
    datasets = {"train": train_dataset, "val": val_dataset}
    dataloaders = {"train": train_loader, "val": val_loader}
    samplers = {"train": train_sampler, "val": val_sampler}

    if not dataloaders["train"] or not dataloaders["val"]:
        raise ValueError("Train or validation set is empty")

    if task == "regression":
        weights = None
        metrics = {
            "train": initialize_regression_metrics(device),
            "val": initialize_regression_metrics(device),
            "test": initialize_regression_metrics(device),
        }
    elif config["task"] == "classification":
        weights = None
        head_structure = config.get("head_structure")
        if head_structure is None:
            raise ValueError("head_structure must be specified in config for multi-head loss")
        
        # Initialize metrics for each head
        metrics = {phase: {} for phase in ["train", "val", "test"]}
        for phase in metrics:
            for head_name, num_classes in head_structure.items():
                metrics[phase][head_name] = initialize_classification_metrics(num_classes, device)            
    else:
        raise ValueError(
            f"Invalid task specified: {task}. Choose 'regression', 'classification' or 'multi_head'."
        )

    best_metrics = {
        "best_loss": float("inf"),
        "best_auc": -float("inf"),  # for classification
        "best_mae": float("inf"),  # for regression
        "best_rmse": float("inf"),  # for regression
    }

    print(
        "Training on examples and validating on :",
        len(dataloaders["train"].dataset),
        len(dataloaders["val"].dataset),
    )

    for epoch in range(epoch_resume, config["num_epochs"]):
        print("Epoch #", epoch)
        for phase in ["train", "val"]:
            start_time = time.time()
            samplers[phase].set_epoch(epoch)

            best_metrics = run_training_or_evaluate_orchestrator(
                model,
                dataloaders[phase],
                datasets[phase],
                phase,
                optimizer,
                scheduler,
                config,
                device,
                task,
                weights,
                metrics,
                best_metrics,
                epoch=epoch,
                run=run,
                labels_map=labels_map,
                scaler=scaler,
                args=args,
            )
            print(f"Epoch {epoch} {phase} time: {time.time() - start_time}")

    if config["run_test"]:
        # Clean up
        dist.destroy_process_group()

        # Initialize best_metrics for each split
        best_metrics = {
            "val": {
                "best_loss": float("inf"),
                "best_auc": -float("inf"),  # for classification
                "best_mae": float("inf"),  # for regression
                "best_rmse": float("inf"),  # for regression
            },
            "test": {
                "best_loss": float("inf"),
                "best_auc": -float("inf"),  # for classification
                "best_mae": float("inf"),  # for regression
                "best_rmse": float("inf"),  # for regression
            },
        }
        for split in ["val", "test"]:
            if do_log:
                # Configure datasets for validation and testing splits
                perform_inference(split, config, False, metrics, best_metrics)

        if args.local_rank == 0:
            wandb.finish()


def sync_tensor_across_gpus(t: torch.Tensor | None) -> torch.Tensor | None:
    """
    Synchronizes a tensor across multiple GPUs in a distributed setting.

    Args:
        t (Union[torch.Tensor, None]): The tensor to be synchronized. If None, nothing happens.

    Returns:
        Union[torch.Tensor, None]: The synchronized tensor, concatenated from all GPUs.
    """
    if t is None or not dist.is_initialized():
        return t

    # Ensure t has at least 1 dimension
    if len(t.shape) == 0:
        t = t.view(1)

    group = dist.group.WORLD
    group_size = dist.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in range(group_size)]

    # Backend compatibility check
    if dist.get_backend() == "nccl":
        t = t.cuda()
    else:
        t = t.cpu()

    dist.all_gather(gather_t_tensor, t)

    return torch.cat(gather_t_tensor, dim=0)


def setup_config(config, transforms, is_master, run):
    """
    Sets up the configuration settings for training or evaluation.

    Args:
        config (dict): The initial configuration settings.
        transforms: The transforms to be applied to the data.
        is_master: A boolean indicating if the current process is the master process.

    Returns:
        config (dict): The updated configuration settings.

    Examples:
        >>> config = {'output_duir': None, 'debug': False, 'test_time_augmentation': False}
        >>> transforms = [Resize(), Normalize()]
        >>> is_master = True
        >>> updated_config = setup_config(config, transforms, is_master)
    """

    config["transforms"] = transforms
    config["debug"] = config.get("debug", False)
    config["test_time_augmentation"] = config.get("test_time_augmentation", False)
    config["weighted_sampling"] = config.get("weighted_sampling", False)
    config["binary_threhsold"] = config.get("binary_threshold", 0.5)

    # Define device
    if is_master:
        run_id = wandb.run.id
        print("Run id", run_id)
        config["run_id"] = run_id
        
        generated_output_dir = generate_output_dir_name(config, run_id=run_id)
        if 'output_dir' in config:
            config["output_dir"] = os.path.join(config["output_dir"], generated_output_dir)
        else:
            config["output_dir"] = generated_output_dir
        pathlib.Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        wandb.config.update(config, allow_val_change=True)
        print("output_folder created", config["output_dir"])
    if config["view_count"] is None:
        print("Loading with 1 view count for mean and std")
        if (
            ("mean" not in config)
            or ("std" not in config)
            or (config["mean"] is None)
            or (config["std"] is None)
        ):
            target_label = config.get("target_label", None)
            if config["task"] == "classification":
                target_label = config.get("head_structure", None)
                
            ## Load a dataset for normalization
            mean, std = orion.utils.get_mean_and_std(
                orion.datasets.Video(
                    root=config["root"],
                    split="train",
                    target_label=target_label,
                    data_filename=config["data_filename"],
                    datapoint_loc_label=config["datapoint_loc_label"],
                    video_transforms=None,
                    resize=config["resize"],
                    weighted_sampling=False,
                    normalize=False,
                ),
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )
            print("Mean", mean)
            print("Std", std)

            config["mean"] = mean
            config["std"] = std
            if is_master:
                wandb.config.update(
                    {
                        "mean": config["mean"],
                        "std": config["std"],
                        "output_dir": config["output_dir"],
                    },
                    allow_val_change=True,
                )
        else:
            print("Using mean and std from config", config["mean"], config["std"])
    else:
        print(f"Loading with {config['view_count']} view count for mean and std")
        if (
            ("mean" not in config)
            or ("std" not in config)
            or (config["mean"] is None)
            or (config["std"] is None)
        ):
            ## Load a dataset for normalization
            mean, std = orion.utils.multi_get_mean_and_std(
                orion.datasets.Video_Multi(
                    root=config["root"],
                    split="train",
                    target_label=config["target_label"],
                    data_filename=config["data_filename"],
                    datapoint_loc_label=config["datapoint_loc_label"],
                    video_transforms=None,
                    resize=config["resize"],
                    weighted_sampling=False,
                    normalize=False,
                ),
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )

            config["mean"] = mean
            config["std"] = std
            if is_master:
                wandb.config.update(
                    {
                        "mean": config["mean"],
                        "std": config["std"],
                        "output_dir": config["output_dir"],
                    },
                    allow_val_change=True,
                )
        else:
            print("Using mean and std from config", config["mean"], config["std"])

    return config


def run_training_or_evaluate_orchestrator(
    model,
    dataloader,
    datasets,
    phase,
    optimizer,
    scheduler,
    config,
    device,
    task,
    weights,
    metrics,
    best_metrics,
    epoch=None,
    run=None,
    labels_map=None,
    scaler=None,
    args=None,
):
    r"""
    Runs the training or evaluation orchestrator for a video model.

    Args:
        model: The video model to train or evaluate.
        dataloader: The dataloader for loading video data.
        datasets: The datasets used for training or evaluation.
        phase (str): The phase of the process, either "train" or "val".
        optimizer: The optimizer for model parameter updates.
        scheduler: The scheduler for adjusting the learning rate.
        config (dict): The configuration settings for training or evaluation.
        device: The device to which the data and model should be moved.
        task (str): The task type, either "regression" or "classification".
        weights: The weights for loss function or metrics.
        metrics (dict): A dictionary containing the metrics to compute.
        best_metrics (dict): A dictionary containing the best metrics achieved so far.
        epoch (int, optional): The current epoch number. Defaults to None.
        run (object, optional): The run object for logging. Defaults to None.
        labels_map (dict, optional): A mapping of label indices to label names. Defaults to None.
        scaler (object, optional): The scaler for data normalization. Defaults to None.

    Returns:
        best_metric: The best metric achieved during training or evaluation.

    Examples:
        >>> model = VideoModel()
        >>> dataloader = DataLoader(video_dataset)
        >>> datasets = {'train': train_dataset, 'val': val_dataset}
        >>> phase = 'train'
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        >>> config = {'loss': 'mse', 'lr': 0.001, 'num_classes': 2}
        >>> device = 'cuda'
        >>> task = 'regression'
        >>> weights = None
        >>> metrics = {'train': {'mae': MAE(), 'mse': MSE()}, 'val': {'mae': MAE(), 'mse': MSE()}}
        >>> best_metrics = {'best_mae': None, 'best_rmse': None, 'b\}
        >>> epoch = 1
        >>> run = wandb.Run()
        >>> labels_map = {
            'Category_1': {0: 'No', 1: 'Yes'}, 
            'Category_2': {0: 'Class_1', 1: 'Class_2', 2: 'Class_3', 3: 'Class_4', 4: 'Class_5'},
            ...
        }
        >>> scaler = StandardScaler()
        >>> best_metric = run_training_or_evaluate_orchestrator(model, dataloader, datasets, phase, optimizer, scheduler, config, device, task, weights, metrics, best_metrics, epoch, run, labels_map, scaler)
    """
    do_log = run is not None  # Run is none if process rank is not 0
    model.train() if phase == "train" else model.eval()
    average_losses, predictions, targets, filenames = train_or_evaluate_epoch(
        model,
        dataloader,
        phase == "train",
        phase,
        optimizer,
        device,
        epoch,
        task,
        save_all=False,
        weights=weights,
        scaler=scaler,
        use_amp=config.get("use_amp", True),
        scheduler=scheduler,
        metrics=metrics,
        config=config,
    )
    y = np.array(targets)
    yhat = np.array(predictions)

    learning_rate = optimizer.param_groups[0]["lr"] if scheduler is not None else config["lr"]

    final_metrics = ""
    if task == "regression":
        print("Epoch logging", epoch)
        # Flatten yhat to 1D if it's 2D
        if yhat.ndim == 3:
            yhat = yhat.reshape(yhat.shape[0], yhat.shape[2])  ## Converts it to 2D Array

        if yhat.ndim == 2 and yhat.shape[1] == 1:
            yhat = yhat.ravel()  # This converts it to a 1D array

        # Example: Update metrics after each batch (within your training loop)
        # preds and target should be PyTorch tensors
        # preds = model(inputs)  # Model predictions
        # target = ...  # Ground truth labels
        final_metrics = metrics[phase].compute()
        metrics[phase].reset()
        auc_score = 0  # Temporary placeholder
        mean_roc_auc_no_nan = 0  # Temporary placeholder

        if do_log:
            log_regression_metrics_to_wandb(phase, final_metrics, average_losses['main_loss'], learning_rate)
            plot_regression_graphics_and_log_binarized_to_wandb(
                y, 
                yhat, 
                phase, 
                epoch, 
                config["binary_threshold"], 
                config
            )

            best_metrics, mae, mse, rmse = update_best_regression_metrics(
                final_metrics, best_metrics
            )

    elif task == "classification":            
        # Log the aggregated loss
        wandb.log({f"{phase}_epoch_loss": average_losses['main_loss']})
        
        # Handle multi-head metrics
        head_structure = config.get("head_structure")
        final_metrics = {}
        mean_roc_auc_no_nan = 0
        total_heads = len(head_structure)
        
        for head_name, num_classes in head_structure.items():
            y = np.array(targets[head_name])
            yhat = np.array(predictions[head_name])
            if num_classes <= 2:
                # Binary classification for this head]
                head_metrics = compute_classification_metrics(metrics[phase][head_name])
                optimal_thresh = compute_optimal_threshold(y, yhat)
                pred_labels = (yhat > optimal_thresh).astype(int)
                if do_log:
                    log_binary_classification_metrics_to_wandb(
                        phase=f"{phase}_{head_name}",
                        loss=average_losses[head_name],
                        auc_score=head_metrics["auc"],
                        optimal_threshold=optimal_thresh,
                        y_true=y,
                        pred_labels=pred_labels,
                        label_map=labels_map.get(head_name) if labels_map else None,
                        learning_rate=learning_rate
                    )
                mean_roc_auc_no_nan += head_metrics["auc"]
            else:
                # Multi-class classification for this head
                head_metrics = compute_multiclass_metrics(metrics[phase][head_name])
                if do_log:
                    log_multiclass_metrics_to_wandb(
                        phase=phase,
                        epoch=epoch,
                        metrics_summary=head_metrics,
                        labels_map=labels_map.get(head_name) if labels_map else None,
                        head_name=head_name,
                        loss=average_losses[head_name],
                        y_true=y,
                        predictions=yhat,
                        learning_rate=learning_rate
                    )
                mean_roc_auc_no_nan += head_metrics["auc_weighted"]
            
            final_metrics[head_name] = head_metrics
        
        # Average AUC across all heads
        mean_roc_auc_no_nan /= total_heads            

    # Update and save checkpoints
    if do_log:
        print(f"best_metrics: {best_metrics}")
        print(f"mean_roc_auc_no_nan: {mean_roc_auc_no_nan}")
        best_loss, best_auc = update_and_save_checkpoints(
            phase,
            epoch,
            average_losses['main_loss'],
            mean_roc_auc_no_nan,
            model,
            optimizer,
            scheduler,
            config["output_dir"],
            wandb,
            best_metrics["best_loss"],
            best_metrics["best_auc"],
            task,
        )
        best_metrics["best_loss"] = best_loss
        best_metrics["best_auc"] = best_auc
        # Generate and save prediction dataframe
        if phase == "val" and (best_metrics["best_loss"] <= average_losses['main_loss']):
            df_predictions = format_dataframe_predictions(
                filenames, predictions, task, config, labels_map, targets
            )
            save_predictions_to_csv(df_predictions, config, phase, epoch)

    return best_metrics


def generate_output_dir_name(config, run_id):
    import time

    """
    Generates a directory name for output based on the provided configuration.

    Args:
        config (dict): The configuration dictionary containing training parameters.
        run_id (str): The ID of the current run.

    Returns:
        str: The generated directory name for saving output.
    """
    # Get current time to create a unique directory name
    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Extract relevant information from the config
    mname = config.get("model_name", "unknown_model")
    bsize = config.get("batch_size", "batch_size")
    run_id = config.get("run_id", "run_id")
    fr = config.get("frames", "frames")
    prd = config.get("period", "period")
    optimizer = config.get("optimizer", "optimizer")
    resume = "resume" if config.get("resume", False) else "new"

    # Create directory name by joining the individual components with underscores
    dir_name = f"{mname}_{bsize}_{fr}_{prd}_{optimizer}_{resume}_{current_time}_{run_id}"

    return dir_name


def update_and_save_checkpoints(
    phase,
    epoch,
    current_loss,
    current_auc,
    model,
    optimizer,
    scheduler,
    output,
    wandb,
    best_loss,
    best_auc,
    task,
):
    """
    Save the current checkpoint. Update the best checkpoint if the current performance is better.

    Args:
        phase (str): The current phase ('train' or 'val').
        epoch (int): The current epoch number.
        current_loss (float): The loss from the current epoch.
        current_auc (float): The AUC from the current epoch, or None if not applicable.
        model (torch.nn.Module): The model to be checkpointed.
        optimizer (torch.optim.Optimizer): The optimizer to be checkpointed.
        scheduler (any scheduler object): The scheduler to be checkpointed.
        output (str): The directory where checkpoints are saved.
        wandb (wandb object): The wandb logging object.
        best_loss (float): The best loss recorded across all epochs for comparison.
        best_auc (float): The best AUC recorded across all epochs for comparison.
        task (str): The current task ('regression' or 'classification').

    Returns:
        tuple: A tuple containing the possibly updated `best_loss` and `best_auc` values.
    """
    # Adjust the paths as needed
    save_path = os.path.join(output, "checkpoint.pt")
    best_path = os.path.join(output, "best.pt")

    # Always save the latest checkpoint
    save_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": current_loss,
        "best_loss": best_loss,
        "auc": current_auc
        if current_auc is not None
        else -1,  # save with -1 if AUC is not applicable
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(save_data, save_path)

    if phase == "val":
        if task == "regression" and current_loss < best_loss:
            best_loss = current_loss
            # Log and save only for regression since AUC is not relevant
            wandb.run.summary["best_loss"] = best_loss
            torch.save(save_data, best_path)
            wandb.log({"best_val_loss": best_loss})
        elif task == "classification":
            if current_loss < best_loss:
                best_loss = current_loss
                wandb.run.summary["best_loss"] = best_loss
                wandb.log({"best_val_loss": best_loss})
            if current_auc > best_auc:
                best_auc = current_auc
                wandb.run.summary["best_auc"] = best_auc
                wandb.log({"best_val_auc": best_auc})

            torch.save(save_data, best_path)

    return best_loss, best_auc


def generate_metrics_log_entry(epoch, phase, loss, conf_mat, roc_auc, start_time, y):
    # Creating log entry for classification metrics
    log_entry = (
        f"{epoch},{phase},{loss},{conf_mat},{roc_auc},{time.time() - start_time},{len(y)}\n"
    )
    return log_entry


def should_update_best_performance(
    current_loss, current_auc, best_loss, best_auc, criterion="loss"
):
    """
    Determine if the current performance is better than the best recorded performance.

    Args:
        current_loss (float): The loss from the current epoch.
        current_auc (float): The AUC from the current epoch.
        best_loss (float): The best loss recorded across all epochs.
        best_auc (float): The best AUC recorded across all epochs.
        criterion (str): The performance criterion to use ('loss' or 'auc').

    Returns:
        bool: True if the current performance is better; False otherwise.
    """
    if criterion == "loss":
        return current_loss < best_loss
    elif criterion == "auc":
        return current_auc > best_auc
    else:
        raise ValueError(f"Invalid criterion specified: {criterion}. Choose 'loss' or 'auc'.")


def build_model(config, device, model_path=None, for_inference=False):
    """
    Build and initialize the model based on the configuration.

    Args:
        config (dict): Configuration parameters for the model.
        device (str): Device to do the training.
        model_path (str, optional): Path to load the model from. Defaults to None.
        for_inference (bool, optional): Whether the model is being built for inference. Defaults to False.

    Returns:
        tuple: Initialized model and other related objects.
    """
    # Check frame and resize parameters
    frames = config.get("frames", 16)  # Default to 16 if not specified
    resize = config.get("resize", 224)  # Default to 224 if not specified
    if config["model_name"].startswith("swin3d"):
        if frames % 2 != 0:
            raise ValueError("swin3d supports only frame counts that are multiples of 2.")
    elif config["model_name"].startswith("x3d"):
        if frames % 8 != 0:
            raise ValueError("x3d models support frame counts that are multiples of 8.")
        if config["model_name"] == "x3d_m" and resize not in [224, 256]:
            raise ValueError("x3d_m supports video values of either 224x224 or 256x256.")
    elif config["model_name"].startswith("mvit"):
        if frames != 16:
            raise ValueError("mvit supports only 16 frames.")

    # Set default resize to 224x224 for models other than x3d_m
    if config["model_name"] not in ["x3d_m", "videopairclassifier"] and resize != 224:
        print(f"Warning: Resize value {resize} is not 224. Setting to default 224x224.")
        config["resize"] = 224

    # Instantiate model based on configuration
    model = load_and_modify_model(config)

    # Set labels_map to None if not defined in the config file
    labels_map = config.get("labels_map", None)
    
    # Add the new classification feature
    if config["task"] == "classification":
        if not labels_map: # TODO: Handle cases where labels_map is None
            raise ValueError("labels_map is not defined in the config file.") # raise error for now until we handle this case
    
    # Check if the model should be resumed or used for inference
    if (model_path and config["resume"]) or for_inference:
        # TODO: Test resuming to ensure correct predictions and training continuation
        print("Device in use:", device)

        # Set a default value for map_location
        map_location = None

        # Load the checkpoint if not in inference mode and a model path is provided
        if not for_inference and model_path:
            print("Loading checkpoint:", model_path)

            # Configure map_location to map to the specific GPU
            map_location = {f"cuda:{0}": f"cuda:{device.index}"}
            print("Map location set to:", map_location)
        elif not model_path:
            # Handle case where model path is None
            model_path = os.path.join(config["output_dir"], "best.pt")

        # Ensure map_location has a value before loading the checkpoint
        if map_location:
            checkpoint = torch.load(model_path, map_location=map_location, weights_only=True)
        else:
            checkpoint = torch.load(model_path, weights_only=True)
        # Uncomment below to debug checkpoint content
        try:
            model_state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))

            # Check and consume "_orig_mod.module." prefix if present
            if any(k.startswith("_orig_mod.module.") for k in model_state_dict.keys()):
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                    model_state_dict, "_orig_mod.module."
                )
                print("Removed prefix '_orig_mod.module.' from state dict")

            # Check and consume "module." prefix if present
            elif any(k.startswith("module.") for k in model_state_dict.keys()):
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                    model_state_dict, "module."
                )
                print("Removed prefix 'module.' from state dict")          

        except RuntimeError as e:
            print(f"Error loading model state dict: {e}")

        model.load_state_dict(model_state_dict)
        print("Model loaded successfully")  
        model.to(device)
        if for_inference == False:
            ### Dont do distributed data parallel if inference is true.
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index], output_device=device.index
            )

        # Additional code to load optimizer and scheduler states, and epoch
        optimizer_state = checkpoint.get("optimizer_state_dict")
        print("Optimizer state loaded successfully")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        print("Scheduler state loaded successfully")
        epoch = checkpoint.get("epoch", 0)
        print("Epoch loaded successfully")
        bestLoss = checkpoint.get("best_loss", float("inf"))
        other_metrics = {
            k: v
            for k, v in checkpoint.items()
            if k
            not in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch"]
        }
        print("Other metrics loaded successfully")

        ## DDP works with TorchDynamo. When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
        # such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes.
        # (See TorchDynamo DDPOptimizer for more information.)

        return model, optimizer_state, scheduler_state, epoch, bestLoss, other_metrics, labels_map
    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index], output_device=device.index
    )
    return model, None, None, 0, float("inf"), {}, labels_map


def load_dataset(split, config, transforms, weighted_sampling):
    """
    Load Video dataset for a given split (train, val, test).

    Args:
        split (str): Split of the dataset to load ('train', 'val', or 'test').
        config (dict): Configuration parameters.
        transforms (dict): PyTorch Vision Transforms
        weighted_sampling: False or vector of weigths according to classes.

    Returns:
        DataLoader: The DataLoader for the specified dataset split.
    """
    missing_fields = []
    if config["mean"] is None:
        missing_fields.append("mean")
    if config["std"] is None:
        missing_fields.append("std")
    if missing_fields:
        raise ValueError(f"Error: The following fields are missing: {', '.join(missing_fields)}")
    else:
        target_label: List[str] = config.get("label_loc_label", None)
        head_structure: Optional[Dict[str, int]] = config.get("head_structure", None)
        if config["task"] == "classification":
            if list(head_structure.keys()) != target_label and split == "train":
                raise ValueError("Error: head_structure does not match target_label")
            
        kwargs = {
            "target_label": target_label,
            "mean": config["mean"],
            "std": config["std"],
            "length": config["frames"],
            "period": config["period"],
            "root": config["root"],
            "data_filename": config["data_filename"],
            "datapoint_loc_label": config.get("datapoint_loc_label", None),
            "apply_mask": config["apply_mask"],
            "resize": config["resize"],
        }

    if split != "inference":
        if config["view_count"] is None:
            dataset = orion.datasets.Video(
                split=split,
                video_transforms=transforms,
                weighted_sampling=weighted_sampling,
                debug=config["debug"],
                **kwargs,
            )
        else:
            dataset = orion.datasets.Video_Multi(
                split=split,
                video_transforms=transforms,
                weighted_sampling=weighted_sampling,
                debug=False,
                **kwargs,
            )
    else:
        if config["view_count"] is None:
            print("Loading video inference dataset")
            dataset = orion.datasets.Video_inference(split=split, **kwargs)
            print("Video inference dataset loaded successfully")
        else:
            dataset = orion.datasets.Video_Multi_inference(split=split, **kwargs)
            print("Video multi inference dataset loaded successfully")

    return dataset


def get_predictions(
    model,
    dataloader,
    device,
    config,
    task,
    use_amp=None,
):
    model.eval()  # Set the model to evaluation mode
    predictions, targets = {}, {}
    filenames = []
    
    # Initialize predictions and targets based on task
    if task == "classification":
        # Initialize for each head in head_structure
        head_structure = config.get("head_structure", {})
        for head_name in head_structure:
            predictions[head_name] = []
            targets[head_name] = []
    else:
        # For single-head tasks, use a single key
        target_label = config.get("labels_map", "main")
        predictions[target_label] = []
        targets[target_label] = []

    with torch.inference_mode():  # Disable gradient calculations
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i, batch in enumerate(dataloader):
                if len(batch) == 3:
                    # If the batch has three elements, it includes the middle tensor
                    data, outcomes, fname = batch
                elif len(batch) == 2:
                    # If the batch has only two elements, it does not include the middle tensor
                    data, fname = batch
                    # Optionally, set outcomes to None or a default value if your logic requires it
                    outcomes = None
                else:
                    raise ValueError("Unexpected batch structure.")

                if config["view_count"] is None:
                    data = data.to(device)
                else:
                    data = torch.stack(data, dim=1)
                    data = data.to(device)

                if config["model_name"] in ["timesformer", "stam"]:
                    data = data.permute(0, 1, 3, 2, 4, 5)

                if config["view_count"] is not None and config["view_count"] >= 1:
                    def flatten(lst):
                        """Recursively flattens a nested list."""
                        if isinstance(lst, list):
                            return [item for sublist in lst for item in flatten(sublist)]
                        else:
                            return [lst]

                    # Flatten the nested list
                    flat_fname = flatten(fname)

                    def reshape_to_pairs(flat_list, group_size=2):
                        """Reshapes a flat list into a list of lists, each containing group_size elements."""
                        return [
                            flat_list[i : i + group_size]
                            for i in range(0, len(flat_list), group_size)
                        ]

                    # Reshape the flattened list into pairs according to the view_count
                    grouped_filenames = reshape_to_pairs(
                        flat_fname, group_size=config["view_count"]
                    )

                    # Append each pair to filenames as a separate row
                    filenames.extend(grouped_filenames)
                else:
                    filenames.extend(fname)

                # Handle targets
                if outcomes is not None and hasattr(outcomes, "detach"):
                    if isinstance(outcomes, dict):
                        # Multi-head case: outcomes is already a dictionary
                        for head_name, head_outcomes in outcomes.items():
                            if head_outcomes.nelement() > 0:
                                targets[head_name].extend(head_outcomes.detach().cpu().numpy())
                    else:
                        # Single-head case: wrap outcome in dictionary
                        target_label = config.get("target_label", "main")
                        if outcomes.nelement() > 0:
                            targets[target_label].extend(outcomes.detach().cpu().numpy())

                # Handle non-4D data if block_size is provided
                if config.get("block_size") is not None and len(data.shape) == 5:
                    batch_size, frames, channels, height, width = data.shape
                    data = data.view(batch_size * frames, channels, height, width)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    outputs = model(data)  # Get model outputs
                    
                    # Convert outputs to dictionary format if it's not already
                    if not isinstance(outputs, dict):
                        target_label = config.get("target_label", "main")
                        outputs = {target_label: outputs}

                    # Process each output based on task and number of classes
                    for head_name, head_outputs in outputs.items():
                        if task == "regression":
                            predictions[head_name].extend(head_outputs.detach().view(-1).cpu().numpy())
                        elif task == "classification":
                            num_classes = head_outputs.shape[1]
                            if num_classes < 2:
                                # Binary classification
                                predictions[head_name].extend(
                                    torch.sigmoid(head_outputs).detach().cpu().numpy()
                                )
                            else:
                                # Multi-class classification
                                predictions[head_name].extend(
                                    torch.softmax(head_outputs, dim=1).detach().cpu().numpy()
                                )
                pbar.update()
                pbar.set_description(f"Processing batch {i+1}/{len(dataloader)}")

    return predictions, targets, filenames


def setup_optimizer_and_scheduler(model, config, epoch_resume=None):
    """
    Sets up the optimizer and scheduler based on the provided configuration.

    Args:
        model: The model for which the optimizer and scheduler are being set up.
        config (dict): Configuration parameters for optimizer and scheduler.

    Returns:
        optim: Configured optimizer.
        scheduler: Configured scheduler (or None if not applicable).
    """
    # Set up optimizer
    if config["optimizer"] == "SGD" or config["optimizer"] is None:
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=0.9,
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "AdamW":
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "RAdam":
        optim = torch.optim.RAdam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    # ... [Include other optimizers like RAdam, AdamW, etc.] ...
    if config["scheduler_type"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=config["factor"],
            patience=config["patience"],
            threshold=config["threshold"],
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )
    elif config["scheduler_type"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, config["lr_step_period"], config["factor"]
        )
    elif config["scheduler_type"] == "cosine_warm_restart":
        print("epoch", epoch_resume)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim,
            T_0=10,
            last_epoch=(epoch_resume - 1) if epoch_resume is not None else -1,
        )
    else:
        scheduler = None
        print("No scheduler specified.")
    # ... [Include other schedulers if needed] ...

    return optim, scheduler


def train_or_evaluate_epoch(
    model,
    dataloader,
    is_training,
    phase,
    optimizer,
    device,
    epoch,
    task,
    save_all=False,
    weights=None,
    scaler=None,
    use_amp=None,
    scheduler=None,
    metrics=None,
    config=None,
):
    model.train(is_training)
    total_losses = {}
    
    # Initialize predictions and targets based on task
    if task == "classification":
        predictions = {}
        targets = {}
    else:
        predictions = []
        targets = []
    
    filenames = []
    model_loss = config.get("loss")
    
    with torch.set_grad_enabled(is_training):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (data, outcomes, fname) in enumerate(dataloader, 1): 
                if config["view_count"] is None:
                    data = data.to(device)
                    
                    # Handle outcomes based on whether it's a dictionary
                    if isinstance(outcomes, dict):
                        outcomes = {k: v.to(device) for k, v in outcomes.items()}
                    else:
                        # If not a dictionary, convert to the expected format
                        target_label = config.get("target_label")
                        if isinstance(target_label, str):
                            target_label = [target_label]
                        outcomes = {target_label[0]: outcomes.to(device)} 

                    if config["model_name"] in ["timesformer", "stam"]:
                        data = data.permute(0, 2, 1, 3, 4)
                else:
                    data = torch.stack(data, dim=1)
                    data = data.to(device)
                    
                    # Handle outcomes based on whether it's a dictionary
                    if isinstance(outcomes, dict):
                        outcomes = {k: v.to(device) for k, v in outcomes.items()}
                    else:
                        # If not a dictionary, convert to the expected format
                        target_label = config.get("target_label")
                        if isinstance(target_label, str):
                            target_label = [target_label]
                        outcomes = {target_label[0]: outcomes.to(device)}

                    if config["model_name"] in ["timesformer", "stam"]:
                        data = data.permute(0, 1, 3, 2, 4, 5)
                filenames.extend(fname)

                if config.get("block_size") is not None and len(data.shape) == 5:
                    data = handle_block_size(data)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    outputs = model(data)
                    
                    # Convert outputs to dictionary format if it's not already
                    if not isinstance(outputs, dict):
                        target_label = config.get("target_label", "main")
                        outputs = {target_label: outputs}
                        
                    # Now returns a dictionnary of losses {"main_loss": main_loss, "loss_1": loss_1, "loss_2": loss_2, ...}    
                    losses = compute_loss_and_update_metrics(
                        outputs,
                        outcomes,
                        task,
                        model_loss,
                        weights,
                        device,
                        metrics,
                        phase,
                        config,
                    )

                # Initialize predictions and targets dictionaries for each head if not already done
                for head_name in outputs.keys():
                    if head_name not in predictions:
                        predictions[head_name] = []
                    if head_name not in targets:
                        targets[head_name] = []
                    
                    # Add predictions and targets for each head
                    predictions[head_name].extend(
                        get_predictions_during_training(outputs[head_name], task)
                    )
                    targets[head_name].extend(outcomes[head_name].detach().cpu().numpy())

                if is_training:
                    backpropagate(optimizer, scaler, losses["main_loss"])
                
                # Add the losses to the total_losses dictionary
                for loss_name, loss_value in losses.items():
                    total_losses[loss_name] = total_losses.get(loss_name, 0.0) + loss_value.item()
                
                # Update the progress bar with the running average of the losses
                pbar.set_postfix(**{loss_name: (total_losses[loss_name] / batch_idx) for loss_name in losses})
                pbar.update()

        # Calculate average losses for the epoch
        average_losses = {loss_name: total_losses[loss_name] / len(dataloader) for loss_name in total_losses}
            
        # Update the scheduler if not training and on validation phase
        if not is_training and phase == "val" and scheduler is not None:
            scheduler.step(average_losses["main_loss"])

    return average_losses, predictions, targets, filenames


def handle_block_size(data):
    batch_size, frames, channels, height, width = data.shape
    data = data.view(batch_size * frames, channels, height, width)
    return data


def compute_loss_and_update_metrics(
    outputs, outcomes, task, model_loss, weights, device, metrics, phase, config
):
    losses = {}
    if task == "regression":
        # Handle dictionary case for regression
        if isinstance(outputs, dict):
            target_label = list(outputs.keys())[0]
            outputs = outputs[target_label]
            outcomes = outcomes[target_label]
            
        outputs = adjust_output_dimensions(outputs)
        loss = LossRegistry.create(model_loss, outputs, outcomes).to(device)
        losses["main_loss"] = loss
        metrics[phase].update(outputs, outcomes)
        
    elif task == "classification":
        # For multi-head loss, outputs and outcomes should be dictionaries
        if not isinstance(outputs, dict) or not isinstance(outcomes, dict):
            raise ValueError("For multi-head loss, outputs and outcomes must be dictionaries")
            
        # Create the multi-head loss function
        head_structure = config.get("head_structure", None)
        loss_structure = config.get("loss_structure", None)
        head_weights = config.get("head_weights", None)
        loss_weights = config.get("loss_weights", None)
        
        if head_structure is None:
            raise ValueError("head_structure must be specified in config for multi-head loss")
        if loss_structure is None:
            raise ValueError("loss_structure must be specified in config for multi-head loss")
        
        multi_head_loss = LossRegistry.create(
            "multi_head",
            head_structure=head_structure,
            loss_structure=loss_structure,
            head_weights=head_weights,
            loss_weights=loss_weights
        ).cuda(device)
        
        # Compute the total loss and individual losses
        loss, individual_losses = multi_head_loss(outputs, outcomes)
        losses["main_loss"] = loss
        losses.update(individual_losses)
        # Update metrics for each head
        for head_name in head_structure.keys():
            num_classes = head_structure[head_name]
            probabilities = get_probabilities(outputs[head_name], {"num_classes": num_classes})
            update_classification_metrics(
                metrics[phase][head_name], 
                probabilities, 
                outcomes[head_name], 
                num_classes
            )

    return losses


def adjust_output_dimensions(outputs):
    if outputs.dim() > 1 and any(size > 1 for size in outputs.shape):
        outputs = outputs.squeeze()
    else:
        outputs = outputs.view(-1)
    return outputs


def get_probabilities(outputs, config):
    if config["num_classes"] <= 2:
        if outputs.ndim > 1 and outputs.shape[1] == 2:
            probabilities = torch.sigmoid(outputs)[:, 1]  # Use the second column for binary classification
        elif outputs.ndim > 1 and outputs.shape[1] == 1:
            probabilities = torch.sigmoid(outputs).squeeze()  # Squeeze the dimension
        else:
            probabilities = torch.sigmoid(outputs)
    else:
        probabilities = torch.softmax(outputs, dim=1)
        if probabilities.dim() > 1:
            probabilities = probabilities.squeeze()
    return probabilities


def get_predictions_during_training(outputs, task):
    if task == "regression":
        return outputs.detach().view(-1).cpu().numpy()
    elif task == "classification":
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze()
            return torch.sigmoid(outputs).detach().cpu().numpy()
        return torch.softmax(outputs, dim=1).detach().cpu().numpy()


def backpropagate(optimizer, scaler, loss):
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def perform_inference(split, config, log_wandb, metrics=None, best_metrics=None):
    """
    Perform inference using the specified model and data based on the provided configuration.

    Args:
        split (str): The split name for inference. If inference, it will not save it to WANDB and run the metrics.
        config (dict): Configuration dictionary containing model and dataset parameters.
        metrics (dict, optional): Dictionary of metrics objects for evaluation. Defaults to None.
        best_metrics (dict, optional): Dictionary of best metrics values. Defaults to None.

    Returns:
        dict: Dictionary of predictions.
    """
    # Build and load the model
    device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    # Use model_path from config, or default to config["output_dir"] if model_path is None
    model_path = config.get("model_path")
    
    task = config.get("task", "regression")
    (
        model,
        optim_state,
        sched_state,
        epoch_resume,
        bestLoss,
        other_metrics,
        labels_map,
    ) = build_model(config, device, model_path=model_path, for_inference=True)
    model.eval()

    # Create data loader for inference
    dataset = load_dataset(split, config, None, False)
    print("Dataset loaded successfully")
    split_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=(config["device"] == "cuda"),
        drop_last=(split != "test" and split != "inference"),
    )

    print("Dataloader created successfully")

    if log_wandb:
        (
            average_losses,
            predictions,
            targets,
            filenames,
        ) = train_or_evaluate_epoch(
            model,
            split_dataloader,
            is_training=False,
            phase=split,
            optimizer=None,
            device=device,
            epoch=epoch_resume,
            task=task,
            save_all=False,
            weights=None,
            scaler=None,
            use_amp=True,
            scheduler=None,
            metrics=metrics,
            config=config,
        )

        # Convert targets and predictions to numpy arrays for subsequent calculations
        targets = np.array(targets)
        predictions = np.array(predictions)

        # Process the output according to the task type
        if task == "regression":
            # Compute final metrics for regression
            final_metrics = metrics[split].compute()
            metrics[split].reset()

            # Update the best metrics and get the current metrics
            best_metrics, mae, mse, rmse = update_best_regression_metrics(
                final_metrics, best_metrics[split]
            )

            # Log regression metrics
            log_regression_metrics_to_wandb(split, final_metrics, average_losses['main_loss'], 0)
            # Plot regression graphics
            plot_regression_graphics_and_log_binarized_to_wandb(
                targets, predictions, split, epoch_resume, config["binary_threshold"], config
            )

        elif task == "classification":   
            # Handle multi-head metrics
            head_structure = config.get("head_structure")
            phase = split
            epoch = epoch_resume
            for head_name, num_classes in head_structure.items():
                y = np.array(targets[head_name])
                yhat = np.array(predictions[head_name])
                if num_classes <= 2:
                    # Binary classification for this head
                    head_metrics = compute_classification_metrics(metrics[phase][head_name])
                    optimal_thresh = compute_optimal_threshold(y, yhat)
                    pred_labels = (yhat > optimal_thresh).astype(int)
                    log_binary_classification_metrics_to_wandb(
                        phase=f"{phase}_{head_name}",
                        loss=average_losses[head_name],
                        auc_score=head_metrics["auc"],
                        optimal_threshold=optimal_thresh,
                        y_true=y,
                        pred_labels=pred_labels,
                        label_map=labels_map.get(head_name) if labels_map else None,
                        learning_rate=0
                    )
                else:
                    # Multi-class classification for this head
                    head_metrics = compute_multiclass_metrics(metrics[phase][head_name])
                    log_multiclass_metrics_to_wandb(
                        phase=phase,
                        epoch=epoch,
                        metrics_summary=head_metrics,
                        labels_map=labels_map.get(head_name) if labels_map else None,
                        head_name=head_name,
                        loss=average_losses[head_name],
                        y_true=y,
                        predictions=yhat,
                        learning_rate=0
                    )
                
    else:
        print("Performing inference only on new dataset. No WANDB logging")
        predictions, targets, filenames = get_predictions(
            model,
            split_dataloader,
            device,
            config,
            task,
            use_amp=True,
        )
    df_predictions = format_dataframe_predictions(
        filenames, predictions, task, config, labels_map, targets
    )

    save_predictions_to_csv(df_predictions, config, split, "inference")

    return df_predictions


def determine_class(y_hat):
    import ast

    # Check if y_hat is a string that looks like a list and convert it
    if isinstance(y_hat, str) and "[" in y_hat and "]" in y_hat:
        y_hat = ast.literal_eval(y_hat.strip())

    # Check if y_hat is a NumPy array and convert it to a list
    if isinstance(y_hat, np.ndarray):
        y_hat = y_hat.tolist()

    # Handle nested lists
    while isinstance(y_hat, list) and len(y_hat) == 1:
        y_hat = y_hat[0]

    # Now y_hat should be a list or a single value
    if isinstance(y_hat, list):
        if len(y_hat) > 1:  # List of probabilities - multi-class classification
            return int(np.argmax(y_hat))
        else:
            raise ValueError(f"List is empty: {y_hat}")

    # Handle the case where y_hat is a single value
    if isinstance(y_hat, (int, float)):
        return int(y_hat > 0.5)
    else:
        raise ValueError(f"Unsupported type or content for y_hat: {y_hat}")

#TODO This function never supported regression output
def format_dataframe_predictions(filenames, predictions, task, config, labels_map, targets):
    """
    Format predictions into a pandas DataFrame.
    
    Args:
        filenames: List of filenames
        predictions: Model predictions (can be single output or dictionary for multi-head)
        task: Task type ('regression', 'classification', or 'multi_head')
        config: Configuration dictionary
        labels_map: Mapping of label indices to label names
        targets: Ground truth values (optional)
        
    Returns:
        pd.DataFrame: Formatted predictions dataframe
    """
    if task == "classification":
        # Initialize DataFrame with filenames
        df_predictions = pd.DataFrame({"filename": filenames})
        
        # Process each head's predictions
        head_structure = config.get("head_structure", {})
        for head_name, num_classes in head_structure.items():

            # Add ground truth if available
            if targets is not None and len(targets[head_name]) > 0:
                df_predictions[f"target_{head_name}"] = np.array(targets[head_name])
            
            # Handle predictions based on number of classes
            head_predictions = np.array(predictions[head_name])
            
            if num_classes < 2:  # Binary classification
                # For binary classification, store raw probabilities
                df_predictions[f"pred_{head_name}"] = head_predictions
                # Store predicted class
                df_predictions[f"{head_name}_class"] = (
                    head_predictions >= config.get("binary_threshold", 0.5)
                ).astype(int)
            else:  # Multi-class classification
                # Store probabilities for each class
                for class_name, idx in labels_map[head_name].items():
                    df_predictions[f"{head_name}_prob_class_{class_name}"] = head_predictions[:, idx]
                # Store predicted class
                df_predictions[f"{head_name}_class"] = np.argmax(head_predictions, axis=1)
                
    elif task == "regression": #TODO doing this for regression for now
        df_predictions = pd.DataFrame({"filename": filenames})
        df_predictions["pred"] = predictions
        df_predictions["target"] = targets

    else:
        raise ValueError(f"Unsupported task: {task}")

    return df_predictions


def save_predictions_to_csv(df_predictions, config, split, epoch):
    import datetime
    import os

    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # Extract the directory name from model_path if it exists
    if config.get("model_path"):
        model_dir = os.path.basename(os.path.dirname(config["model_path"]))
    else:
        model_dir = os.path.basename(config["output_dir"])

    filename = f"{split}_predictions_epoch_{epoch}.csv"
    best_filename = f"{split}_predictions_epoch_best.csv"

    # Construct the output path
    output_path = os.path.join(config["output_dir"], filename)
    best_output_path = os.path.join(config["output_dir"], best_filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving predictions to {output_path}")
    df_predictions.to_csv(output_path)

    print(f"Saving best predictions to {best_output_path}")
    df_predictions.to_csv(best_output_path)


def create_transforms(config):
    transform_list = []
    if config.get("transforms") is not None:
        for item in config["transforms"]:
            transform_name = item["transform"]
            params = item.get("params", {})

            # Handle null values correctly
            params = {
                k: (None if isinstance(v, np.ndarray) and pd.isnull(v).all() else v)
                for k, v in params.items()
            }

            # Dynamically get the transform class from torchvision.transforms
            transform_class = getattr(transforms, transform_name)

            # Create an instance of the transform class with the provided parameters
            transform_instance = transform_class(**params)
            transform_list.append(transform_instance)
    return transform_list


def setup_run(args, config_defaults):
    """
    Set up the run for training or evaluation.

    Args:
        args: The command line arguments.
        config_defaults: The default configuration values.

    Returns:
        The run object for logging with wandb or None if not running on rank 0.
    """
    if args.local_rank == 0:
        run = wandb.init(
            entity=config_defaults["entity"],
            project=config_defaults["project"],
            config=config_defaults,
            name=config_defaults["project"],  # Assuming 'tag' is an attribute of args
            resume=config_defaults.get("resume", False),
            id=config_defaults.get("wandb_id", None),
            allow_val_change=True,
        )
    else:
        run = None
    return run


def infer_and_convert(value):
    """
    Attempt to convert a string value to a float, an integer, or a boolean as appropriate.
    If all conversions fail, return the original string.
    """
    # Convert to boolean if applicable
    if value.lower() in ["true", "false"]:
        return value.lower() == "true"

    # Try to convert to float first (to handle decimal values)
    try:
        float_val = float(value)
        # If the float value is equivalent to an int, return it as an int (for whole numbers)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        pass

    # Return the original string if all conversions fail
    return value


def update_config(config, additional_args):
    for key, value in additional_args.items():
        converted_value = infer_and_convert(value)
        config[key] = converted_value  # Update the configuration with the converted value
    return config


def main():
    """
    Entry point of the program.
    Parses command line arguments, loads configuration file, updates configuration with additional arguments,
    sets up logging, creates transforms, and executes the main training and evaluation process.
    """
    import os

    import yaml

    args, additional_args = orion.utils.arg_parser.parse_args()
    print("Arguments:", args)
    print("Additional Arguments:", additional_args)

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file '{args.config_path}' not found.")

    with open(args.config_path) as file:
        config_defaults = yaml.safe_load(file)
        print("Initial config", config_defaults)
        # print("Initial config", config_defaults)
    # Check if the script is running in sweep mode
    # Initialize a WandB run if logging, otherwise return None

    if additional_args is not None:
        config_defaults = update_config(config_defaults, additional_args)
        print("Updated config", config_defaults)
    else:
        print("No additional arguments")

    run = setup_run(args, config_defaults) if args.local_rank == 0 else None

    # Create the transforms
    transform_list = create_transforms(config_defaults)

    # Run the main training and evaluation process
    execute_run(config_defaults=config_defaults, transforms=transform_list, args=args, run=run)


if __name__ == "__main__":
    main()
