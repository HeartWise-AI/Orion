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
from sklearn import metrics as sk_metrics
from sklearn import preprocessing
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

import wandb

dir2 = os.path.abspath("/volume/Orion/orion")
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path:
    sys.path.append(dir1)

import tqdm

import orion
from orion.datasets import Video
from orion.models import movinet, pyvid_multiclass_x3d, stam, timesformer, vivit, x3d, x3d_multi
from orion.models.videopairclassifier import VideoPairClassifier
from orion.utils import arg_parser, dist_eval_sampler, plot, video_training_and_eval
from orion.utils.plot import (
    bootstrap_metrics,
    bootstrap_multicalss_metrics,
    compute_multiclass_metrics,
    initialize_regression_metrics,
    log_regression_metrics_to_wandb,
    metrics_from_moving_threshold,
    plot_moving_thresh_metrics,
    plot_multiclass_confusion,
    plot_multiclass_rocs,
    plot_preds_distribution,
    plot_regression_graphics,
    update_best_regression_metrics,
)

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

torch.set_float32_matmul_precision("high")


def execute_run(config_defaults=None, transforms=None, args=None, run=None):
    torch.cuda.empty_cache()
    # Check to see if local_rank is 0
    is_master = args.local_rank == 0
    print("is_master", is_master)

    config = setup_config(config_defaults, transforms, is_master)
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

    # Model building and training setup
    (
        model,
        optim_state,
        sched_state,
        epoch_resume,
        bestLoss,
        other_metrics,
        labels_map,
    ) = build_model(config, device, model_path=config.get("model_path"))

    ## DDP works with TorchDynamo. When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
    # such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes.
    # (See TorchDynamo DDPOptimizer for more information.)
    model = torch.compile(model)

    # watch gradients only for rank 0
    if is_master:
        run.watch(model)
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

    # Set up dataloaders for weighted_sampling
    if config["weighted_sampling"] == True:
        print("Using weighted sampling")
        train_sampler = WeightedRandomSampler(
            train_dataset.weight_list,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=train_sampler,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
    else:
        print("Using updated train sampler distributed")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=train_sampler,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=val_sampler,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

    dataloaders = {"train": train_loader, "val": val_loader}
    datasets = {"train": train_dataset, "val": val_dataset}
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
        class_weights = config["class_weights"]
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

    best_metrics = {
        "best_loss": float("inf"),
        "best_auc": -float("inf"),  # for classification
        "best_mae": float("inf"),  # for regression
        "best_rmse": float("inf"),  # for regression
        "best_r2": float("inf"),  # for regression
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
            )
            print(f"Epoch {epoch} {phase} time: {time.time() - start_time}")

    if config["run_test"]:
        for split in ["val", "test"]:
            # Configure datasets for validation and testing splits

            dataset_loader, dataset = load_data(split, config, None, False)

            # DataLoader for the current split
            split_dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                shuffle=False,
                pin_memory=(device.type == "cuda"),
            )

            # Perform evaluation on the current split
            (
                split_loss,
                split_yhat,
                split_y,
                filenames,
            ) = orion.utils.video_training_and_eval.train_or_evaluate_epoch(
                model,
                split_dataloader,
                False,
                split,
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
                config=config,
            )

            # Convert targets and predictions to numpy arrays for subsequent calculations
            split_y = np.array(split_y)
            split_yhat = np.array(split_yhat)

            # Process the output according to the task type
            if task == "regression":
                if do_log:
                    split_y = np.array(split_y)
                    split_yhat = np.array(split_yhat)
                    mae, mse, rmse, r2 = plot_regression_metrics(split_y, split_yhat)
                    log_regression_metrics_to_wandb(split, split_loss, mae, mse, rmse, r2)
            elif task == "classification":
                if config["num_classes"] <= 2:
                    # Binary classification
                    (
                        auc_score,
                        optimal_thresh,
                        conf_mat,
                        pred_labels,
                    ) = calculate_binary_classification_metrics(split_y, split_yhat)

                    print(auc_score)
                    print(optimal_thresh)
                    log_binary_classification_metrics_to_wandb(
                        split,
                        epoch,
                        split_loss,
                        auc_score,
                        optimal_thresh,
                        split_y,
                        pred_labels,
                        labels_map,
                        do_log=do_log,
                    )
                else:
                    # Multiclass classification
                    metrics_summary = calculate_multiclass_metrics(
                        split_y, split_yhat, config["num_classes"]
                    )
                    log_multiclass_metrics_to_wandb(
                        split,
                        epoch,
                        metrics_summary,
                        labels_map,
                        split_y,
                        split_yhat,
                        do_log=do_log,
                    )

                    # Extract roc_auc from summary and calculate mean ROC AUC, excluding the 'micro' average and NaNs
                    roc_auc = metrics_summary["roc_auc"]
                    roc_auc_minus_micro = {k: v for k, v in roc_auc.items() if k != "micro"}
                    roc_auc_minus_micro_no_nan = {
                        k: v for k, v in roc_auc_minus_micro.items() if not np.isnan(v)
                    }
                    mean_roc_auc_no_nan = np.mean(list(roc_auc_minus_micro_no_nan.values()))
                    print(
                        f"{split} mean ROC AUC score after removing NaNs: {mean_roc_auc_no_nan}"
                    )

            # Save prediction results
            if do_log:
                df_predictions = pd.DataFrame(
                    list(zip(filenames, split_y, split_yhat)),
                    columns=["filename", "y_true", "y_hat"],
                )
                df_predictions.to_csv(os.path.join(config["output"], f"{split}_predictions.csv"))

        # Clean up
        dist.destroy_process_group()


def setup_config(config, transforms, is_master):
    """
    Sets up the configuration settings for training or evaluation.

    Args:
        config (dict): The initial configuration settings.
        transforms: The transforms to be applied to the data.
        is_master: A boolean indicating if the current process is the master process.

    Returns:
        config (dict): The updated configuration settings.

    Examples:
        >>> config = {'output': None, 'debug': False, 'test_time_augmentation': False}
        >>> transforms = [Resize(), Normalize()]
        >>> is_master = True
        >>> updated_config = setup_config(config, transforms, is_master)
    """

    config["transforms"] = transforms
    config["output"] = config["output"] or generate_output_dir_name(config)
    print("Output folder", config["output"])
    config["debug"] = config.get("debug", False)
    config["test_time_augmentation"] = config.get("test_time_augmentation", False)
    config["weighted_sampling"] = config.get("weighted_sampling", False)
    config["binary_threhsold"] = config.get("binary_threshold", 0.5)

    # Define device
    if is_master:
        pathlib.Path(config["output"]).mkdir(parents=True, exist_ok=True)
        print("output_folder created", config["output"])

    if config["view_count"] is None:
        print("Loading with 1 view count for mean and std")

        mean, std = orion.utils.get_mean_and_std(
            orion.datasets.Video(
                root=config["root"],
                split="train",
                target_label=config["target_label"],
                data_filename=config["data_filename"],
                datapoint_loc_label=config["datapoint_loc_label"],
                video_transforms=transforms,
                resize=config["resize"],
                weighted_sampling=False,
            ),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        print("Mean", mean)
        print("Std", std)
        config["mean"] = mean
        config["std"] = std

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
):
    """
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
        >>> best_metrics = {'best_mae': None, 'best_rmse': None, 'best_r2': None}
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
        print(final_metrics)
        metrics[phase].reset()
        auc_score = 0  # Temporary placeholder
        mean_roc_auc_no_nan = 0  # Temporary placeholder

        if do_log:
            log_regression_metrics_to_wandb(phase, final_metrics, loss, learning_rate)

            plot_regression_graphics(y, yhat, phase, epoch, config["binary_threshold"], config)

            best_metrics = update_best_regression_metrics(final_metrics, best_metrics)

    elif task == "classification":
        if config["num_classes"] <= 2:
            final_metrics = compute_classification_metrics(metrics[phase])
            print(final_metrics)
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
            metrics_summary = calculate_multiclass_metrics(y, yhat, config["num_classes"])
            log_multiclass_metrics_to_wandb(
                phase, epoch, metrics_summary, labels_map, y, yhat, learning_rate, do_log=do_log
            )
            # Update the best metrics for multi-class classification, handle logic for your use case
            roc_auc = metrics_summary["roc_auc"]

            # Get the dictionary you mentioned, excluding 'micro' which is average itself
            roc_auc_minus_micro = {k: v for k, v in roc_auc.items() if k != "micro"}

            roc_auc_minus_micro_no_nan = {
                k: v for k, v in roc_auc_minus_micro.items() if not np.isnan(v)
            }

            # Now compute the mean ROC AUC after the NaN values are removed
            mean_roc_auc_no_nan = np.mean(list(roc_auc_minus_micro_no_nan.values()))
            print(f"Mean ROC AUC score after removing NaNs: {mean_roc_auc_no_nan}")

    # Update and save checkpoints
    best_loss, best_auc = update_and_save_checkpoints(
        phase,
        epoch,
        loss,
        mean_roc_auc_no_nan,
        model,
        optimizer,
        scheduler,
        config["output"],
        wandb,
        best_metrics["best_loss"],
        best_metrics["best_auc"],
        task,
        do_log=do_log,
    )
    best_metrics["best_loss"] = best_loss
    best_metrics["best_auc"] = best_auc

    # Generate and save prediction dataframe
    if phase == "val":
        if do_log:
            print(y)
            print(yhat)
            df_predictions = pd.DataFrame(
                list(zip(filenames, y, yhat.squeeze())), columns=["filename", "y_true", "y_hat"]
            )
            df_predictions.to_csv(os.path.join(config["output"], "df_predictions.csv"))

    return best_metrics


def generate_output_dir_name(config):
    import time

    """
    Generates a directory name for output based on the provided configuration.

    Args:
        config (dict): The configuration dictionary containing training parameters.

    Returns:
        str: The generated directory name for saving output.
    """
    # Get current time to create a unique directory name
    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Extract relevant information from the config
    mname = config.get("model_name", "unknown_model")
    bsize = config.get("batch_size", "batch_size")
    fr = config.get("frames", "frames")
    prd = config.get("period", "period")
    optimizer = config.get("optimizer", "optimizer")
    resume = "resume" if config.get("resume", False) else "new"

    # Create directory name by joining the individual components with underscores
    dir_name = f"{mname}_{bsize}_{fr}_{prd}_{optimizer}_{resume}_{current_time}"

    return dir_name


def initialize_classification_metrics(num_classes, device):
    """
    Initializes and returns a collection of classification metrics based on the number of classes.

    Args:
        num_classes (int): The number of classes.
        device: The device to which the metrics should be moved.

    Returns:
        metrics (dict): A dictionary containing the initialized classification metrics.

    Examples:
        >>> num_classes = 2
        >>> device = 'cuda:0'
        >>> metrics = initialize_classification_metrics(num_classes, device)"""

    if num_classes <= 2:
        # Binary Classification Metrics
        metrics = {
            "auc": torchmetrics.AUROC(task="binary").to(device),
            "confmat": torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device),
        }
    else:
        # Multi-Class Classification Metrics
        metrics = {
            "auc": torchmetrics.AUROC(task="multilabel", num_classes=num_classes).to(device),
            "auc_weighted": torchmetrics.AUROC(
                task="multiclass", num_classes=num_classes, average="weighted"
            ).to(device),
            "confmat": torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=num_classes
            ).to(device),
        }
    return metrics


def update_classification_metrics(metrics, preds, target, num_classes):
    """
    Updates the classification metrics based on the predictions and targets.

    Args:
        metrics (dict): A dictionary containing the classification metrics to update.
        preds (torch.Tensor): The predicted values.
        target (torch.Tensor): The target values.
        num_classes (int): The number of classes.

    Examples:
        >>> metrics = {'auc': AUC(), 'confmat': ConfusionMatrix()}
        >>> preds = torch.tensor([0.8, 0.2, 0.6])
        >>> target = torch.tensor([1, 0, 1])
        >>> num_classes = 2
        >>> update_classification_metrics(metrics, preds, target, num_classes)"""

    if num_classes <= 2:
        # Update for Binary Classification
        metrics["auc"].update(preds, target.int())
        metrics["confmat"].update((preds > 0.5).int(), target.int())
    else:
        # Update for Multi-Class Classification
        # Convert predictions to label format
        print(preds)
        if preds.ndim > 1 and preds.shape[1] == 1:
            pred_labels = preds.flatten()
        print(preds)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes)
        metrics["auc"].update(preds, target_one_hot)
        metrics["auc_micro"].update(preds, target_one_hot)
        metrics["confmat"].update(preds.argmax(dim=1), target)


def compute_classification_metrics(metrics):
    """
    Computes classification metrics based on the provided metrics dictionary.

    Args:
        metrics (dict): A dictionary containing the metrics to compute.

    Returns:
        computed_metrics (dict): A dictionary containing the computed classification metrics.

    Examples:
        >>> metrics = {'accuracy': Accuracy(), 'precision': Precision()}
        >>> computed_metrics = compute_classification_metrics(metrics)
    """

    computed_metrics = {}
    for metric_name, metric in metrics.items():
        computed_value = metric.compute()

        # Check if tensor has a single element
        if computed_value.numel() == 1:
            computed_metrics[metric_name] = computed_value.item()
        else:
            # Convert to a floating point tensor before taking the mean
            computed_value_float = computed_value.type(torch.float32)
            computed_metrics[metric_name] = computed_value_float.mean().item()

    for metric in metrics.values():
        metric.reset()

    return computed_metrics


def compute_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_index = tpr - fpr
    optimal_idx = np.argmax(j_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def log_binary_classification_metrics_to_wandb(
    phase,
    epoch,
    loss,
    auc_score,
    optimal_threshold,
    y_true,
    pred_labels,
    label_map,
    learning_rate=None,
    do_log=False,
):
    if do_log:
        # Log binary classification metrics to WandB
        wandb.log({f"{phase}_epoch_loss": loss})
        wandb.log({f"{phase}_AUC": auc_score})
        wandb.log({f"{phase}_optimal_thresh": optimal_threshold})
        # Convert predictions to label format
        print(pred_labels)
        if pred_labels.ndim > 1 and pred_labels.shape[1] == 1:
            pred_labels = pred_labels.flatten()
        print(pred_labels)

        # Define your class names (replace with actual class names)
        class_names = [str(label) for label in label_map]

        # Log the confusion matrix in wandb
        wandb.log(
            {
                f"{phase}_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=y_true, preds=pred_labels, class_names=class_names
                )
            }
        )

        if learning_rate is not None:
            wandb.log({"learning_rate": learning_rate})


def log_multiclass_metrics_to_wandb(
    phase,
    epoch,
    metrics_summary,
    labels_map,
    y_true,
    predictions,
    learning_rate=None,
    do_log=False,
):
    """Log multi-class metrics to wandb.

    Args:
        phase (str): The phase of the evaluation (e.g., "train", "val").
        epoch (int): The epoch number.
        metrics_summary (dict): A dictionary containing the computed metrics summary.
        labels_map (dict): A dictionary mapping class indices to class labels.
        y_true (numpy.ndarray): The true labels.
        predictions (numpy.ndarray): The predicted probabilities.

    Raises:
        None

    Returns:
        None

    """

    # Retrieve metrics
    fpr = metrics_summary["fpr"]
    tpr = metrics_summary["tpr"]
    roc_auc = metrics_summary["roc_auc"]
    conf_mat = metrics_summary["conf_mat"]
    # print("conf_mat", conf_mat)

    # Compute y_pred from predictions
    y_pred = np.argmax(predictions, axis=1)
    # print(y_pred)

    # Prepare ROC curve plot for multi-class classification
    fig, ax = plt.subplots()
    colors = cycle(
        [
            "blue",
            "green",
            "red",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "purple",
            "pink",
            "lightblue",
            "lightgreen",
            "gray",
        ]
    )

    for i, color in zip(labels_map, colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve for class {labels_map[i]} (area = {roc_auc[i]:0.2f})",
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-Class ROC curve for {phase} phase, epoch {epoch}")
    plt.legend(loc="lower right")
    # Log the AUC for each class and overall micro averaged AUC
    if do_log:
        for i in labels_map.keys():
            wandb.log({f"{phase}_roc_auc_class_{labels_map[i]}": roc_auc[i]})

        if learning_rate is not None:
            wandb.log({"learning_rate": learning_rate})

            # ...

        # For micro averaged AUC
        wandb.log({f"{phase}_roc_auc_micro": roc_auc["micro"]})
        # Log the ROC curve and confusion matrix with whttps://vscode-remote+attached-002dcontainer-002b7b22636f6e7461696e65724e616d65223a222f61676974617465645f77696c6c69616d73227d-0040ssh-002dremote-002b10-002e128-002e228-002e213.vscode-resource.vscode-cdn.net/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_ranking.py:1133andb

    # Create a remapping dictionary from your original labels to new labels
    remapping_dict = {
        original_label: new_label
        for new_label, original_label in enumerate(sorted(labels_map.keys()))
    }
    # Remap your original labels_map to match new labels
    labels_map_remap = {
        new_label: labels_map[original_label]
        for original_label, new_label in remapping_dict.items()
    }

    # Remap y_true and y_pred using the remapping dictionary
    y_true_remap = np.vectorize(remapping_dict.get)(y_true)

    # Safeguard the remapping to handle values not in remapping_dict
    remap_function = lambda x: remapping_dict.get(x, -1)
    y_pred_remap = np.vectorize(remap_function)(y_pred.squeeze())  # Ensure y_pred is a 1D array

    if not all([el is not None for el in y_pred]):
        print("y_pred contains None values")

    if not all([el is not None for el in y_true]):
        print("y_true contains None values")
    if do_log:
        wandb.log(
            {
                f"{phase}_roc_curve_epoch_{epoch}": wandb.Image(fig),
                # f"{phase}_conf_mat_epoch_{epoch}": wandb.plot.confusion_matrix(
                #    probs=None, y_true=y_true_remap, preds=y_pred_remap, class_names=list(labels_map_remap.values()))
            }
        )


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
    do_log=False,
):
    if do_log:  ##Only if its the main process
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
            "state_dict": model.state_dict(),
            "loss": current_loss,
            "auc": current_auc
            if current_auc is not None
            else -1,  # save with -1 if AUC is not applicable
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(save_data, save_path)

        if phase == "val":
            if task == "regression" and current_loss < best_loss:
                best_loss = current_loss
                # Log and save only for regression since AUC is not relevant
                wandb.run.summary["best_loss"] = best_loss
                torch.save(save_data, best_path)
                wandb.log({"best_val_loss": best_loss})
            elif task == "classification" and current_loss < best_loss:
                best_loss = current_loss
                best_auc = current_auc
                # Log and save for classification since both loss and AUC are relevant
                wandb.run.summary["best_loss"] = best_loss
                wandb.run.summary["best_auc"] = best_auc
                torch.save(save_data, best_path)
                wandb.log({"best_val_loss": best_loss, "best_val_auc": best_auc})

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


def build_model(config, device, model_path=None):
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
        dataset = pd.read_csv(
            os.path.join("../../data/", config["data_filename"]),
            sep="µ",
            engine="python",
        )
        # uses label and int_label column from datasetfor wandb plotting
        labels_map = dict(
            zip(
                dataset[config["target_label"]],
                dataset[config["label_loc_label"]],
            )
        )
        labels_map = dict(sorted(labels_map.items(), key=lambda item: item[1]))
        print("Labels map", labels_map)
    else:
        labels_map = None

    # Loading pretrained weights if specified
    if model_path and config["resume"] == True:
        ##TO DO : TEST RESUMING AND MAKE SURE YOU GET THE RIGHT PREDICTIONS AND TRAINING
        print("Loading checkpoint", model_path)
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % device.index}
        print("Map location", map_location)

        checkpoint = torch.load(model_path, map_location=map_location)
        # print("Model ddp", checkpoint)

        model_state_dict = checkpoint["state_dict"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            model_state_dict, "_orig_mod.module."
        )

        model.load_state_dict(model_state_dict)
        model.to(device)

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


## ROBERT : TO REMOVE
def load_data_for_inference(config):
    """
    Create a DataLoader for the Video dataset specifically for inference using the provided configuration.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        torch.utils.data.DataLoader: DataLoader ready for inference.
    """
    dataset = orion.datasets.Video(
        root=config["root"],
        data_filename=config["data_filename"],
        datapoint_loc_label=config["datapoint_loc_label"],
        split=config["split"],
        video_transforms=config.get("transforms", None),
        target_label="y_pred_cat",
        mean=config["mean"],
        std=config["std"],
        resize=config["resize"],
    )

    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Check the file paths and CSV formatting.")

    loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    return loader


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
    kwargs = {
        "target_label": config["target_label"],
        "mean": config["mean"],
        "std": config["std"],
        "length": config["frames"],
        "period": config["period"],
        "root": config["root"],
        "data_filename": config["data_filename"],
        "datapoint_loc_label": config["datapoint_loc_label"],
        "apply_mask": config["apply_mask"],
        "resize": config["resize"],
        "model_name": config["model_name"],
    }

    if config["view_count"] is None:
        dataset = orion.datasets.Video(
            split=split,
            video_transforms=transforms,
            weighted_sampling=weighted_sampling,
            debug=config["debug"],
            **kwargs,
        )
    else:
        dataset = orion.datasets.Echo_Multi(
            split=split,
            video_transforms=transforms,
            weighted_sampling=weighted_sampling,
            debug=False,
            **kwargs,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=(split == "train"),
        pin_memory=(config["device"] == "cuda"),
        drop_last=split != "test",
    )
    return dataloader, dataset


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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim,
            verbose=True,
            last_epoch=epoch_resume,
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
    with torch.set_grad_enabled(is_training):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}") as pbar:
            for batch in dataloader:
                data, outcomes, fname = batch
                data, outcomes = data.to(device), outcomes.to(device)
                filenames.extend(fname)

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
                    # print("Use amp", use_amp)
                    # Get model outputs, handle block_size within output generation

                    outputs = model(data)

                    # Loss computation
                    if task == "regression":
                        outputs = outputs.squeeze()
                        metrics[phase].update(outputs, outcomes)
                        loss = compute_regression_loss(outputs, outcomes, model_loss).cuda(device)
                    elif task == "classification":
                        num_dims = outputs.dim()
                        # If outputs is 2D and the last dimension is 1 (e.g., [batch_size, 1]), squeeze the last dimension
                        if num_dims == 2 and outputs.size(1) == 1:
                            outputs = outputs.squeeze(-1)
                        update_classification_metrics(
                            metrics[phase], outputs, outcomes, config["num_classes"]
                        )

                        # print(outputs)
                        # print("Pre squeeze", outputs.shape)

                        # print("Post squeeze", outputs.shape)

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
                    if outputs.dim() == 1:
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

            if is_training == False:
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


def perform_inference(config):
    """
    Perform inference using the specified model and data based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing model and dataset parameters.

    Returns:
        dict: Dictionary of predictions.
    """
    # Build and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, device, model_path=config["chkpt_path"])
    model.to(device)
    model.eval()

    # Create data loader for inference
    dataloader = load_data_for_inference(config)

    # Perform inference
    predictions = {}
    with torch.no_grad():
        for X, _, filenames in dataloader:
            X = X.to(device)
            outputs = model(X)

            for filename, output in zip(filenames, outputs):
                predictions[filename] = output.cpu().numpy()

    return predictions


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
    if args.local_rank == 0:
        run = wandb.init(
            entity=config_defaults["entity"],
            project=config_defaults["project"],
            config=config_defaults,
            name=config_defaults["project"],  # Assuming 'tag' is an attribute of args
            resume=config_defaults.get("resume", False),
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
    import os

    import yaml

    args, additional_args = orion.utils.arg_parser.parse_args()
    print("Arguments:", args)
    print("Additional Arguments:", additional_args)

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file '{args.config_file}' not found.")

    with open(args.config_file) as file:
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
