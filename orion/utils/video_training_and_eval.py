import os
import pathlib
import sys
import time
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms

import wandb

dir2 = os.path.abspath("/volume/Orion/orion")
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path:
    sys.path.append(dir1)

import tqdm

import orion
from orion.datasets import Video
from orion.models import (
    movinet,
    pyvid_multiclass_x3d,
    stam,
    timesformer,
    vivit,
    x3d,
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    do_log = run is not None  # Run is none if process rank is not 0

    # set the device
    total_devices = torch.cuda.device_count()
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
    print(model_path)
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

    print("Model path:", model_path)
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

    ## DDP works with TorchDynamo. When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
    # such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes.
    # (See TorchDynamo DDPOptimizer for more information.)
    model = torch.compile(model)

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
    train_loader, train_dataset = load_data(
        "train", config, transforms, config["weighted_sampling"]
    )
    val_loader, val_dataset = load_data("val", config, transforms, config["weighted_sampling"])

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
        num_classes = 2 if config["num_classes"] <= 2 else config["num_classes"]
        metrics = {
            "train": initialize_classification_metrics(num_classes, device),
            "val": initialize_classification_metrics(num_classes, device),
            "test": initialize_classification_metrics(num_classes, device),
        }
        class_weights = config.get("class_weights", None)

        if config["class_weights"] is None:
            print("Not using weighted sampling and not using class weights specified in config")
            weights = None
        elif config["class_weights"] == "balanced_weights":
            labels = train_dataset.outcome
            weights = torch.tensor(1 - (np.bincount(labels) / len(labels)), dtype=torch.float32)
            print("Weights", weights)
            weights = weights.to(device)
        else:
            print("Using class weights specified in config", class_weights)
            weights = torch.tensor(class_weights, dtype=torch.float32)
            weights = weights.to(device)
    else:
        raise ValueError(
            f"Invalid task specified: {task}. Choose 'regression' or 'classification'."
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
                perform_inference(split, config, metrics, best_metrics)

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
        if ("output_dir" not in config) or (config["output_dir"] is None):
            config["output_dir"] = generate_output_dir_name(config, run_id=run_id)
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
            ## Load a datswet for normalization
            mean, std = orion.utils.get_mean_and_std(
                orion.datasets.Video(
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
        >>> labels_map = {0: 'cat', 1: 'dog'}
        >>> scaler = StandardScaler()
        >>> best_metric = run_training_or_evaluate_orchestrator(model, dataloader, datasets, phase, optimizer, scheduler, config, device, task, weights, metrics, best_metrics, epoch, run, labels_map, scaler)
    """
    do_log = run is not None  # Run is none if process rank is not 0
    model.train() if phase == "train" else model.eval()
    loss, predictions, targets, filenames = train_or_evaluate_epoch(
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
            log_regression_metrics_to_wandb(phase, final_metrics, loss, learning_rate)

            plot_regression_graphics_and_log_binarized_to_wandb(
                y, yhat, phase, epoch, config["binary_threshold"], config
            )

            best_metrics, mae, mse, rmse = update_best_regression_metrics(
                final_metrics, best_metrics
            )

    elif task == "classification":
        if config["num_classes"] <= 2:
            final_metrics = compute_classification_metrics(metrics[phase])
            optimal_thresh = compute_optimal_threshold(y, yhat)
            pred_labels = (yhat > optimal_thresh).astype(int)

            log_binary_classification_metrics_to_wandb(
                phase,
                epoch,
                loss,
                final_metrics["auc"],
                optimal_thresh,
                y,
                pred_labels,
                labels_map,
                learning_rate,
                do_log=do_log,
            )

            mean_roc_auc_no_nan = final_metrics["auc"]
        else:
            metrics_summary = compute_multiclass_metrics(metrics[phase])
            log_multiclass_metrics_to_wandb(
                phase, epoch, metrics_summary, labels_map, y, yhat, learning_rate, do_log=do_log
            )
            # Update the best metrics for multi-class classification, handle logic for your use case
            mean_roc_auc_no_nan = metrics_summary["auc_weighted"]
            print(f"Mean ROC AUC score after removing NaNs: {mean_roc_auc_no_nan}")
    print("final metrics", final_metrics)
    # Update and save checkpoints
    if do_log:
        best_loss, best_auc = update_and_save_checkpoints(
            phase,
            epoch,
            loss,
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
        print("best metrics", best_metrics)
        # Generate and save prediction dataframe
        if phase == "val":
            df_predictions = format_dataframe_predictions(
                filenames, yhat.squeeze(), task, phase, config, y
            )
            save_predictions_to_csv(df_predictions, config, phase)

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
                print("best loss", current_loss)
                best_loss = current_loss
                wandb.run.summary["best_loss"] = best_loss
                wandb.log({"best_val_loss": best_loss})
            if current_auc > best_auc:
                print("best auc", best_auc)
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

    Returns:
        torch.nn.Module: Initialized model.
    """
    # Instantiate model based on configuration
    print(config["task"])
    print(config["num_classes"])

    if config["model_name"] == "x3d":
        model = x3d(num_classes=config["num_classes"], task=config["task"])
    elif config["model_name"] == "x3d_legacy":
        print("Using legacy X3d model with extra regression fc2 layer")
        model = x3d_legacy(num_classes=config["num_classes"], task=config["task"])
    elif config["model_name"] == "pyvid_multiclass_x3d":
        model = pyvid_multiclass_x3d(num_classes=config["num_classes"], resize=config["resize"])
    elif config["model_name"] == "timesformer":
        model = timesformer(num_classes=config["num_classes"], resize=config["resize"])
    elif config["model_name"] == "stam":
        model = stam(num_classes=config["num_classes"], resize=config["resize"])
    elif config["model_name"] == "vivit":
        model = vivit(
            num_classes=config["num_classes"],
            resize=config["resize"],
            num_frames=config["frames"],
        )
    elif config["model_name"] in [
        "c2d_r50",
        "i3d_r50",
        "slow_r50",
        "slowfast_r50",
        "slowfast_r101",
        "slowfast_16x8_r101_50_50",
        "csn_r101",
        "r2plus1d_r50",
        "x3d_xs",
        "x3d_s",
        "x3d_m",
        "x3d_l",
        "mvit_base_16x4",
        "mvit_base_32x3",
        "efficient_x3d_xs",
        "efficient_x3d_s",
    ]:
        from orion.models import pytorchvideo_model

        model = pytorchvideo_model(
            config["model_name"], config["num_classes"], task=config["task"]
        )
    elif config["model_name"] in ["swin3d_s", "swin3d_b"]:
        from orion.models import get_fmodel, pytorchvideo_model

        # model = get_fmodel(config["model_name"])
        model = pytorchvideo_model(config["model_name"], config["num_classes"], config["task"])
    else:
        print("Error: Model name not found :", config["model_name"])

    # Add the new classification feature
    if config["task"] == "classification":
        try:
            dataset = pd.read_csv(
                os.path.join("../../data/", config["data_filename"]),
                sep="µ",
                engine="python",
            )
            # Initialize labels_map with None for each class index
            labels_map = {i: None for i in range(config["num_classes"])}

            # Update labels_map with actual labels from dataset
            for int_label, label in zip(
                dataset[config["target_label"]], dataset[config["label_loc_label"]]
            ):
                labels_map[int(int_label)] = label

            print("Labels map before sorting:", labels_map)

            # Optionally, sort the labels_map if needed
            labels_map = dict(sorted(labels_map.items(), key=lambda item: item[0]))

            print("Labels map after sorting:", labels_map)
        except:
            print("No label map")
            labels_map = None
    else:
        labels_map = None

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
            checkpoint = torch.load(model_path, map_location=map_location)
        else:
            checkpoint = torch.load(model_path)
        # Uncomment below to debug checkpoint content
        # print("Model checkpoint content:", checkpoint)
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

            model.load_state_dict(model_state_dict)
        except RuntimeError as e:
            print(f"Error loading model state dict: {e}")

        model.load_state_dict(model_state_dict)
        model.to(device)
        if for_inference == False:
            ### Dont do distributed data parallel if inference is true.
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index], output_device=device.index
            )

        # Additional code to load optimizer and scheduler states, and epoch
        optimizer_state = checkpoint.get("optimizer_state_dict")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        epoch = checkpoint.get("epoch", 0)
        bestLoss = checkpoint.get("best_loss", float("inf"))
        other_metrics = {
            k: v
            for k, v in checkpoint.items()
            if k
            not in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch"]
        }

        ## DDP works with TorchDynamo. When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
        # such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes.
        # (See TorchDynamo DDPOptimizer for more information.)

        return model, optimizer_state, scheduler_state, epoch, bestLoss, other_metrics, labels_map
    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index], output_device=device.index
    )
    return model, None, None, 0, float("inf"), {}, labels_map


def load_data(split, config, transforms, weighted_sampling):
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
    if config["mean"] is None or config["std"] is None:
        print("Error: 'mean' or 'std' values are missing.")
        return None, None
    else:
        kwargs = {
            "target_label": config.get("target_label", None),
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
        dataset = orion.datasets.Video_inference(split=split, **kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=(split == "train"),
        pin_memory=(config["device"] == "cuda"),
        drop_last=(split != "test" and split != "inference"),
    )
    return dataloader, dataset


def get_predictions(
    model,
    dataloader,
    device,
    config,
    task,
    use_amp=None,
):
    model.eval()  # Set the model to evaluation mode
    predictions, targets, filenames = [], [], []

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

                data = data.to(device)
                filenames.extend(fname)
                if outcomes is not None and hasattr(outcomes, "detach"):
                    # Check if outcomes is not empty
                    if outcomes.nelement() > 0:
                        targets.extend(outcomes.detach().cpu().numpy())
                # Handle non-4D data if block_size is provided
                if (
                    config.get("block_size") is not None and len(data.shape) == 5
                ):  # assuming shape is (B, T, C, H, W)
                    batch_size, frames, channels, height, width = data.shape
                    data = data.view(batch_size * frames, channels, height, width)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    outputs = model(data)  # Get model outputs
                    # Process outputs for regression or classification tasks
                    if task == "regression":
                        predictions.extend(outputs.detach().view(-1).cpu().numpy())
                    elif task == "classification":
                        if outputs.dim() <= 2:
                            # For binary classification, add an extra dimension to make it [batch_size, 1]
                            predictions.extend(
                                torch.sigmoid(outputs).detach().cpu().numpy()
                            )  ## Add sigmoid to get 0 -1 predictions
                        else:
                            predictions.extend(
                                torch.softmax(outputs, dim=1).detach().cpu().numpy()
                            )  # Softmax for classification

                pbar.update()  # Ensure this is correctly indented
                pbar.set_description(
                    f"Processing batch {i+1}/{len(dataloader)}"
                )  # Optional: set a manual description

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
            verbose=False,
            # ... [Other scheduler parameters] ...
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
            verbose=True,
            last_epoch=(epoch_resume - 1),
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
    total_loss = 0.0
    total_items = 0
    predictions, targets, filenames = [], [], []
    model_loss = config.get("loss")
    print("is_training", is_training)
    with torch.set_grad_enabled(is_training):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch}") as pbar:
            for batch in dataloader:
                # print(device)
                data, outcomes, fname = batch
                data, outcomes = data.to(device), outcomes.to(device)
                filenames.extend(fname)

                # print("Data device:", data.device)
                # print("Is data empty?", data.nelement() == 0)

                # Handle non-4D data if block_size is provided
                if (
                    config.get("block_size") is not None and len(data.shape) == 5
                ):  # assuming shape is (B, T, C, H, W)
                    # Flatten the temporal and batch dimension to make data 4D
                    batch_size, frames, channels, height, width = data.shape
                    data = data.view(batch_size * frames, channels, height, width)

                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    # Get model outputs, handle block_size within output generation

                    outputs = model(data)

                    # Loss computation
                    if task == "regression":
                        # print("Outputs first", outputs)
                        # print("Outcomes", outcomes)
                        # print("fname", fname)
                        # Conditional squeeze: only apply if there are more than 1 dimension
                        if outputs.dim() > 1 and any(size > 1 for size in outputs.shape):
                            outputs = outputs.squeeze()
                        else:
                            outputs = outputs.view(-1)
                        metrics[phase].update(outputs, outcomes)
                        loss = compute_regression_loss(outputs, outcomes, model_loss).cuda(device)
                    elif task == "classification":
                        num_dims = outputs.dim()

                        if config["num_classes"] <= 2:
                            # For binary classification
                            if outputs.dim() > 1:
                                outputs = outputs.squeeze()
                            probabilities = F.sigmoid(outputs)
                        else:
                            probabilities = F.softmax(outputs, dim=1)

                        update_classification_metrics(
                            metrics[phase], probabilities, outcomes, config["num_classes"]
                        )

                        loss = compute_classification_loss(
                            outputs, outcomes, model_loss, weights
                        ).cuda(device)

                # Record metrics for each output, handle regression and classification separately
                if task == "regression":
                    predictions.extend(
                        outputs.detach().view(-1).cpu().numpy()
                    )  # For regression, keep predictions unprocessed
                elif task == "classification":
                    # Assuming 'outputs' is your raw model output
                    if config["num_classes"] <= 2:
                        if outputs.dim() > 1:
                            outputs = outputs.squeeze()
                        # For binary classification, add an extra dimension to make it [batch_size, 1]
                        predictions.extend(
                            F.sigmoid(outputs).detach().cpu().numpy()
                        )  ## Add sigmoid to get 0 -1 predictions
                    else:
                        predictions.extend(
                            F.softmax(outputs, dim=1).detach().cpu().numpy()
                        )  # Softmax for classification

                # Training logic

                if is_training:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item() * data.size(0)
                total_items += data.size(0)
                targets.extend(outcomes.detach().cpu().numpy())

                pbar.set_postfix(loss=(total_loss / total_items))
                pbar.update()

            if (is_training == False) & (phase == "val") & (scheduler is not None):
                # Assuming scheduler is ReduceLROnPlateau, which requires validation loss
                scheduler.step(loss)

        average_loss = total_loss / total_items

    return average_loss, predictions, targets, filenames


# Define additional helper functions for computing loss
def compute_regression_loss(outputs, targets, model_loss):
    """
    Computes the regression loss based on the specified model loss.

    Args:
        outputs (Tensor): The predicted outputs from the model.
        targets (Tensor): The target values.
        model_loss (str): The type of model loss to use. Supported options are "mse", "huber", "l1_loss", and "rmse" for regression and "bce_loss" or "ce_loss" for classificaiton.

    Returns:
        Tensor: The computed regression loss.

    Raises:
        NotImplementedError: If the specified model loss is not implemented.

    Examples:
        >>> outputs = torch.tensor([0.5, 0.8, 1.2])
        >>> targets = torch.tensor([1.0, 1.5, 2.0])
        >>> model_loss = "mse"
        >>> compute_regression_loss(outputs, targets, model_loss)
        tensor(0.1667)
    """

    if model_loss == "mse":
        return torch.nn.functional.mse_loss(outputs.view(-1), targets)
    elif model_loss == "huber":
        return torch.nn.functional.huber_loss(outputs.view(-1), targets, delta=0.10)
    elif model_loss == "l1_loss":
        return torch.nn.functional.l1_loss(outputs.view(-1), targets)
    elif model_loss == "rmse":
        return torch.sqrt(torch.nn.functional.mse_loss(outputs.view(-1), targets))
    else:
        raise NotImplementedError(f"Loss type '{model_loss}' not implemented.")


def compute_classification_loss(outputs, targets, model_loss, weights):
    if model_loss == "bce_logit_loss":
        criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
    elif model_loss == "ce_loss":
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        outputs = outputs.squeeze()
        targets = targets.squeeze().long()
    else:
        raise NotImplementedError(f"Loss type '{model_loss}' not implemented.")

    return criterion(outputs, targets)


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
    print("model path", model_path)

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
    split_dataloader, dataset = load_data(split, config, None, False)

    if log_wandb:
        (
            split_loss,
            split_yhat,
            split_y,
            filenames,
        ) = orion.utils.video_training_and_eval.train_or_evaluate_epoch(
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
        split_y = np.array(split_y)
        split_yhat = np.array(split_yhat)

        # Process the output according to the task type
        if task == "regression":
            # Compute final metrics for regression
            # print("Current phase:", split)
            # print("Available keys in metrics:", metrics.keys())
            final_metrics = metrics[split].compute()
            metrics[split].reset()

            # Update the best metrics and get the current metrics
            best_metrics, mae, mse, rmse = update_best_regression_metrics(
                final_metrics, best_metrics[split]
            )

            # Log regression metrics
            log_regression_metrics_to_wandb(split, final_metrics, split_loss, 0)
            # Plot regression graphics
            plot_regression_graphics_and_log_binarized_to_wandb(
                split_y, split_yhat, split, epoch_resume, config["binary_threshold"], config
            )

        elif task == "classification":
            if config["num_classes"] <= 2:
                final_metrics = compute_classification_metrics(metrics[split])
                optimal_thresh = compute_optimal_threshold(split_y, split_yhat)
                pred_labels = (split_yhat > optimal_thresh).astype(int)
                # Binary classification logging
                log_binary_classification_metrics_to_wandb(
                    split,
                    epoch_resume,
                    split_loss,
                    final_metrics["auc"],
                    optimal_thresh,
                    split_y,
                    pred_labels,
                    labels_map,
                    do_log=True,
                )

            else:
                metrics_summary = compute_multiclass_metrics(metrics[split])

                # Multiclass classification logging
                log_multiclass_metrics_to_wandb(
                    split,
                    epoch_resume,
                    metrics_summary,
                    labels_map,
                    split_y,
                    split_yhat,
                    0,
                    do_log=True,
                )
    else:
        print("Performing inference only on new dataset. No WANDB logging")
        split_yhat, split_y, filenames = get_predictions(
            model,
            split_dataloader,
            device,
            config,
            task,
            use_amp=True,
        )

    df_predictions = format_dataframe_predictions(
        filenames, split_yhat, task, split, config, split_y
    )

    save_predictions_to_csv(df_predictions, config, split)

    return df_predictions


def determine_class(y_hat):
    import ast

    # Check if y_hat is a string that looks like a list and convert it
    if isinstance(y_hat, str) and "[" in y_hat and "]" in y_hat:
        y_hat = ast.literal_eval(y_hat.strip())

    # Check if y_hat is a NumPy array and convert it to a list
    if isinstance(y_hat, np.ndarray):
        y_hat = y_hat.tolist()

    # Now y_hat should be a list or a single value
    if isinstance(y_hat, list):
        if len(y_hat) == 1:  # Single probability - binary classification
            y_hat = y_hat[0]  # Extract the single value from the list
        elif len(y_hat) > 1:  # List of probabilities - multi-class classification
            return int(np.argmax(y_hat))
        else:
            raise ValueError(f"List is empty: {y_hat}")

    # Handle the case where y_hat is a single value
    if isinstance(y_hat, (int, float)):
        return int(y_hat > 0.5)
    else:
        raise ValueError(f"Unsupported type or content for y_hat: {y_hat}")


def format_dataframe_predictions(filenames, split_yhat, task, split, config, split_y=None):
    # The above code is checking if the variable `split_y` is not equal to `None`. If it is not
    # `None`, then the code inside the `if` statement will be executed.

    if split_y is not None and len(split_y) > 0:
        # If split_y is provided, include it in the DataFrame
        data = list(zip(filenames, split_y, split_yhat))
        df_predictions = pd.DataFrame(data, columns=["filename", "y_true", "y_hat"])
    else:
        # If split_y is None, create DataFrame without the y_true column
        data = list(zip(filenames, split_yhat))
        df_predictions = pd.DataFrame(data, columns=["filename", "y_hat"])

    if task == "classification":
        try:
            df_predictions["y_hat"] = df_predictions["y_hat"].apply(
                lambda x: np.fromstring(x.strip("[]"), sep=" ")
            )
        except:
            print("Error converting string to array. Check if the y_hat column is a string.")
        df_predictions["argmax_class"] = df_predictions["y_hat"].apply(determine_class)
    else:
        binary_threshold = config["binary_threshold"]
        df_predictions["argmax_class"] = df_predictions["y_hat"].apply(
            lambda x: 1 if x >= binary_threshold else 0
        )

    return df_predictions


def save_predictions_to_csv(df_predictions, config, split):
    import datetime

    current_date = datetime.datetime.now().strftime("%Y%m%d")

    model_dir = os.path.basename(
        os.path.dirname(config.get("model_path") or config["output_dir"])
    )
    filename = f"{model_dir}_{split}_predictions_{current_date}.csv"
    output_path = os.path.join(config["output_dir"], filename)

    # create the output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    print(f"Saving predictions to {output_path}")
    df_predictions.to_csv(output_path)


def create_transforms(config):
    transform_list = []
    for item in config["transforms"]:
        transform_name = item["transform"]
        params = item.get("params", {})

        # Handle null values correctly
        params = {k: (None if v == "null" else v) for k, v in params.items()}

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
