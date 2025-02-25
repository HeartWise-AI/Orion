"""
Functions for plotting results
"""

import logging
import os
import random
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
import tqdm
from sklearn import metrics
from sklearn.metrics import auc, confusion_matrix, roc_curve

import wandb

logging.getLogger("PIL").setLevel(logging.WARNING)  # hide DEBUG messages from PIL
logging.getLogger("wandb").setLevel(logging.WARNING)  # hide DEBUG messages from wandb
logging.getLogger("urllib3").setLevel(logging.WARNING)  # hide DEBUG messages from urllib3
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # hide DEBUG messages from matplotlib


def plot_loss(log_path):
    """
    INPUT
    log_path (path): path to log.csv file to plot
    """
    sns.set_style("darkgrid")

    log = pd.read_csv(log_path, skiprows=1, header=None, error_bad_lines=False, sep=";")
    log = log[~log[0].str.contains("Starting")]
    log = log[~log[0].str.contains("Resuming")]
    log = log[~log[0].str.contains("Best")]
    log = log[~log[0].str.contains("crop")]
    split_log = log[0].str.split(",", expand=True)

    split_log.rename(columns={0: "epoch", 1: "type", 2: "loss"}, inplace=True)
    split_log["epoch"] = split_log["epoch"].astype(int)
    split_log["type"] = split_log["type"].astype(str)
    split_log["loss"] = split_log["loss"].astype(float)

    fig = sns.lineplot(data=split_log, x="epoch", y="loss", hue="type")
    fig.set(xticks=split_log.epoch[2::8])


def plot_loss_and_lr(log_path):
    """This function designed to read new-style multiclass log csvs
    INPUT
    log_path (path): path to log.csv file to plot
    """
    sns.set_style("darkgrid")

    log = pd.read_csv(log_path, skiprows=1, header=None, error_bad_lines=False, sep=";")
    # remove unwanted rows
    log = log[~log[0].str.contains("Starting")]
    log = log[~log[0].str.contains("Resuming")]
    log = log[~log[0].str.contains("Best")]
    log = log[~log[0].str.contains("crop")]

    # get rows containing loss values
    loss_log = log[(log[0].str.contains("val")) | (log[0].str.contains("train"))]
    loss_log = loss_log[0].str.split(",", expand=True)

    loss_log.rename(columns={0: "epoch", 1: "type", 2: "loss"}, inplace=True)
    loss_log["epoch"] = loss_log["epoch"].astype(int)
    loss_log["type"] = loss_log["type"].astype(str)
    loss_log["loss"] = loss_log["loss"].astype(float)
    loss_log["index"] = loss_log.index.to_numpy()

    # pdb.set_trace()

    # get rows containing LR values
    lrs = log[log[0].str.contains("]]")][0].str.split(",", expand=True)
    lrs = lrs.drop_duplicates(subset=[lrs.shape[1] - 1])
    lr_df = pd.DataFrame(
        zip(lrs[lrs.shape[1] - 1], lrs.index.to_numpy() + 1), columns=["lr", "index"]
    )

    # left merge
    final = loss_log.merge(lr_df, how="left", on="index")

    # fill in lrs
    final["lr"].fillna(method="ffill", inplace=True)
    final["lr"].fillna(method="bfill", inplace=True)
    final["lr"] = final["lr"].astype(float)

    # Make plot
    fig, ax1 = plt.subplots()
    # twin axis for overplotting
    ax2 = ax1.twinx()

    sns.lineplot(data=final, x="epoch", y="loss", hue="type", ax=ax1)
    sns.lineplot(data=final[final["type"] == "train"], x="epoch", y="lr", ax=ax2, color="black")


def plot_multiclass_rocs(y, yhat, encoder, save_dir=None, tag="test"):
    """Plots ROC curves for multiclass classfication

    Args:
        y (array): True labels, one-hot encoded, size(N samples, N classes)
        yhat (array): Model predictions, one-hot encoded, size(N samples, N classes)
        encoder (scikitlearn one hot encoder): scikitlearn one hot encoder
        save_dir (path): path to save
    """
    sns.set_style("darkgrid")

    num_classes = len(encoder.get_feature_names_out())

    roc_fig, ax = plt.subplots(figsize=(10, 10))

    # switch for binary vs multiclass classification
    if num_classes == 2:
        fpr, tpr, _ = metrics.roc_curve(y, yhat)
        roc_auc = metrics.auc(fpr, tpr)
        print(f"AUROC is {np.round(roc_auc,3)}")
        plt.plot(fpr, tpr)
    else:
        for i in range(num_classes):
            fpr, tpr, _ = metrics.roc_curve(y[:, i], yhat[:, i])
            roc_auc = metrics.auc(fpr, tpr)
            print(f"AUROC for class {i} = {np.round(roc_auc,3)}")
            plt.plot(fpr, tpr, linestyle=random.choice(["-", "--", ":", "-."]))

    roc_fig.legend(
        [i.rsplit("_")[1] for i in encoder.get_feature_names_out()],
        loc="right",
        title="Class",
    )
    plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
    ax.set_title("ROC Curves, " + tag.upper())

    if save_dir is not None:
        title = "val_ROCs.pdf" if tag != "test" else tag + "_ROCs.pdf"
        plt.savefig(os.path.join(save_dir, title))


def plot_multiclass_confusion(conf_mat, save_dir=None, tag="test"):
    """Plots confusion matrix for multiclass classification

    Args:
        conf_mat (array): confusion matrix array, size (N classes, N classes)
        save_dir (path): path to save
    """
    confusion_plot = plt.figure(figsize=(10, 10))

    ax = plt.subplot()
    sns.set(font_scale=1.1)
    sns.heatmap(conf_mat, annot=True, ax=ax, cmap="Blues", fmt="g")

    # labels, title and ticks
    ax.set_xlabel("Predicted labels", fontsize=20)
    ax.set_ylabel("Observed labels", fontsize=20)
    ax.set_title("Confusion Matrix, " + tag.upper(), fontsize=23)

    if save_dir is not None:
        if tag != "test":
            title = "val_confusion_matrix.pdf"
        else:
            title = tag + "_confusion_matrix.pdf"
        plt.savefig(os.path.join(save_dir, title))

    plt.show()


def plot_regression_graphics_and_log_binarized_to_wandb(
    y, yhat, phase, epoch, binary_threshold, config
):
    """
    Plot regression-related metrics and log a binarized version of the data (via WandB).
    Specifically, we treat the targets (y) and predictions (yhat) as regression outputs,
    but also apply a threshold (`binary_threshold`) to evaluate them in a binary sense
    (e.g., y > threshold => 1, else 0).

    Parameters
    ----------
    y : array-like, torch.Tensor, or dict
        Ground truth values. May be:
          - Numpy array
          - Torch tensor
          - Dict with a single key (e.g., {"Value": [...]})
    yhat : array-like, torch.Tensor, or dict
        Predicted values in the same shapes/formats as y.
    phase : str
        "train", "val", etc. to indicate the phase for logging.
    epoch : int
        Current epoch number (for logging).
    binary_threshold : float
        Threshold above which we call predictions (or targets) "1", else "0".
    config : dict
        Configuration dictionary that should include 'metrics_control' specifying
        how to optimize the threshold, etc.

    Raises
    ------
    ValueError
        If either y or yhat is a dict with multiple keys,
        since we only support single-key dicts for this function.
    """

    import numpy as np
    import sklearn
    import torch

    from orion.utils.plot import (
        metrics_from_moving_threshold,
        plot_moving_thresh_metrics,
        plot_preds_distribution,
    )

    # --- 1) Unwrap y if it is a dictionary or a numpy array containing a dictionary ---
    if isinstance(y, dict):
        # If it contains exactly one key (e.g., 'Value'), extract it
        if len(y) == 1:
            y = y[next(iter(y.keys()))]  # Extract the value associated with the first key
        else:
            raise ValueError(
                f"Expected 1 key in `y` dict for regression, got keys: {list(y.keys())}"
            )
    elif isinstance(y, np.ndarray) and isinstance(y.item(), dict):
        # Handle the case where y is a numpy array containing a dictionary
        y_dict = y.item()
        if len(y_dict) == 1:
            y = y_dict[next(iter(y_dict.keys()))]
        else:
            raise ValueError(
                f"Expected 1 key in `y` dict for regression, got keys: {list(y_dict.keys())}"
            )

    # --- 2) Unwrap yhat if it is a dictionary or a numpy array containing a dictionary ---
    if isinstance(yhat, dict):
        # If it contains exactly one key (e.g., 'Value'), extract it
        if len(yhat) == 1:
            yhat = yhat[next(iter(yhat.keys()))]
        else:
            raise ValueError(
                f"Expected 1 key in `yhat` dict for regression, got keys: {list(yhat.keys())}"
            )
    elif isinstance(yhat, np.ndarray) and isinstance(yhat.item(), dict):
        # Handle the case where yhat is a numpy array containing a dictionary
        yhat_dict = yhat.item()
        if len(yhat_dict) == 1:
            yhat = yhat_dict[next(iter(yhat_dict.keys()))]
        else:
            raise ValueError(
                f"Expected 1 key in `yhat` dict for regression, got keys: {list(yhat_dict.keys())}"
            )

    # --- 3) Convert torch tensors to numpy arrays ---
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    if torch.is_tensor(yhat):
        yhat = yhat.detach().cpu().numpy()

    # --- 4) Ensure final y and yhat are numpy arrays (in case they're lists) ---
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(yhat, np.ndarray):
        yhat = np.array(yhat)

    # --- 5) Binarize the ground-truth, based on the provided threshold ---
    y_cat = np.where(y > binary_threshold, 1, 0)

    # The rest is unchanged: compute an ROC, find an optimal threshold for predictions, etc.
    metric_for_cutoff_locator = config["metrics_control"]["optim_thresh"]

    # Optionally plot distribution of predicted probabilities
    if config["metrics_control"]["plot_pred_distribution"]:
        plot_preds_distribution(y, yhat, phase=phase, epoch=epoch)
        metrics_moving_thresh = metrics_from_moving_threshold(y_cat, yhat)
        optim_thresh = metrics_moving_thresh[metric_for_cutoff_locator].idxmax()
        print(f"Optimal cut-off threshold: {optim_thresh}, based on {metric_for_cutoff_locator}")

        if config["metrics_control"]["plot_metrics_moving_thresh"]:
            plot_moving_thresh_metrics(
                metrics_moving_thresh,
                optim_thresh=optim_thresh,
                phase=phase,
                epoch=epoch,
            )

    # --- 6) Compute and log ROC / confusion matrix ---
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_cat, yhat)
    roc_auc = auc(fpr, tpr)

    import wandb

    if phase == "val":
        wandb.log(
            {
                "val_roc_auc": roc_auc,
                "val_conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_cat,
                    preds=(yhat > optim_thresh),
                    title=f"val, epoch {epoch}",
                ),
            },
            commit=False,
        )
        data = [[float(a), float(b)] for (a, b) in zip(y, yhat)]
        table = wandb.Table(data=data, columns=["y", "yhat"])
        plotid = f"{epoch}_scatterplot_val"
        wandb.log(
            {
                plotid: wandb.plot.scatter(
                    table, "y", "yhat", title=f"Scatter plot val, epoch {epoch}"
                )
            }
        )
    elif phase == "train":
        wandb.log({"train_roc_auc": roc_auc}, commit=False)


def compute_roc_auc(y_true, y_hat):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        * Compute ROC AUC for binary or multi-class
    ---------------------------------------------------------------------------------
    Args:
        * y_true
            - numpy array, if multi-class, should be of shape n x c, n- number of samples, c- number of classes
        * y_hat
            - numpy, format same as y_true
    ---------------------------------------------------------------------------------
    Returns:
        * A scalar if binary class
        A dictionary if multi-class
    ---------------------------------------------------------------------------------
    """
    assert y_true.shape == y_hat.shape
    nd = y_true.ndim
    if nd == 1:  # binary class
        fpr, tpr, _ = metrics.roc_curve(y_true, y_hat)
        roc_auc = metrics.auc(fpr, tpr)
    else:
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(y_true.shape[1]):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_hat[:, i])
            roc_auc[f"roc_auc_{i}"] = metrics.auc(fpr[i], tpr[i])

    return roc_auc


def metrics_from_binvecs(y_true_bin, y_hat_bin):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        * Compute a series of metrics for binary classification performance
    ---------------------------------------------------------------------------------
    Args:
        * y_true_bin
            - flat numpy array
        * b
            - array_like: second argument to `func`
    ---------------------------------------------------------------------------------
    Returns:
        * A dictionary of metrics
            - p:
    ---------------------------------------------------------------------------------
    """
    # sanity check, input arrays should be 1D
    assert y_true_bin.ndim == 1
    assert y_hat_bin.ndim == 1

    # sanity check, if input arrays are binary
    assert np.array_equal(y_true_bin, y_true_bin.astype(bool))
    assert np.array_equal(y_hat_bin, y_hat_bin.astype(bool))

    y_true_bin = y_true_bin.astype(bool)
    y_hat_bin = y_hat_bin.astype(bool)

    """
    Suppress numpy 'division by zero' warnings to avoid flooding the console, since it's expected.
    This warning suppression behavior is limited to the lines within the 'with' block below only,
    and should not affect other parts.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # sanity check, "division by zero" should exhibit the following behavior
        assert np.float64(1.0) / np.float64(0.0) == np.inf
        assert np.isnan(np.float64(0.0) / np.float64(0.0))

        p = y_true_bin.sum().astype(np.float64)
        n = (y_true_bin.size - p).astype(np.float64)
        tp = np.logical_and(y_true_bin, y_hat_bin).sum().astype(np.float64)  # true positive
        tn = (
            np.bitwise_not(np.logical_or(y_true_bin, y_hat_bin)).sum().astype(np.float64)
        )  # true negative, simplified by De Morgan's law
        fp = (
            np.logical_and(np.bitwise_not(y_true_bin), y_hat_bin).sum().astype(np.float64)
        )  # false positive
        fn = (
            np.logical_and(y_true_bin, np.bitwise_not(y_hat_bin)).sum().astype(np.float64)
        )  # false negative

        sensitivity = tp / p
        specificity = tn / n
        ppv = tp / (tp + fp)  # positive predictive value, precision
        npv = tn / (tn + fn)  # negative predictive value
        dor = (tp * tn) / (fp * fn)  # diagnostic odds ratio

        f1_score = 2 * tp / (2 * tp + fp + fn)
        g_mean = np.sqrt(sensitivity * specificity)
        youdens_index = sensitivity + specificity - 1

    return {
        "p": p,
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "dor": dor,
        "f1_score": f1_score,
        "g_mean": g_mean,
        "youdens_index": youdens_index,
    }


def metrics_from_moving_threshold(y_true, y_hat, nsteps=100):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        * Compute metrics with a moving threshold. The output dataframe can be used to determine the
        optimal threshold for binary classification and plotting purpose.
    ---------------------------------------------------------------------------------
    Args:
        * y_true
            - 1D numpy array, should contain only binary values (0,1 or bool)
        * y_hat
            - numpy array, predictions by the model, should contain probability values within range [0,1]
        * nsteps
            - int, how many steps to move the cut-off threshold within range [0,1]
    ---------------------------------------------------------------------------------
    Returns:
        * A dataframe containing the metrics
            index: cut-off threshold
            columns: metrics
    ---------------------------------------------------------------------------------
    """

    metrics_moving_thresh = {}
    for i in range(nsteps):
        thresh = float(i) / nsteps
        y_hat_bin = np.where((y_hat >= thresh), 1, np.where((y_hat < thresh), 0, np.nan))
        metrics_moving_thresh[thresh] = metrics_from_binvecs(y_true, y_hat_bin)

    metrics_moving_thresh = pd.DataFrame.from_dict(metrics_moving_thresh, orient="index")
    metrics_moving_thresh.index.name = "threshold"

    return metrics_moving_thresh


def bootstrap(a, b, func, samples=1000):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        * Computes a bootstrapped confidence intervals for ``func(a, b)''
    ---------------------------------------------------------------------------------
    Args:
        * a
            - array_like: first argument to `func`
        * b
            - array_like: second argument to `func`
        * func
            - callable: Function to compute confidence intervals for. ``dataset[i][0]''
                    is expected to be the i-th video in the dataset, which should be a
                    ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        * samples
            - int, optional (1000):  Number of samples to compute
    ---------------------------------------------------------------------------------
    Returns:
        * A tuple of (`func(a, b)`, estimated 5-th percentile, estimated 95-th percentile)
    ---------------------------------------------------------------------------------
    Sanity Checks / Bug fixes:
        * None
    ---------------------------------------------------------------------------------
    Called By:
        * None
    ---------------------------------------------------------------------------------
    TODO:
        * None
    """
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for _ in range(samples):
        ind = np.random.choice(len(a), len(a))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps)

    return (
        func(a, b),
        bootstraps[round(0.05 * len(bootstraps))],
        bootstraps[round(0.95 * len(bootstraps))],
    )


def bootstrap_metrics(y_true_bin, y_hat, thresh, nsample=1000, q=0.05):
    """
    ---------------------------------------------------------------------------------
       Purpose:
           * Compute bootstrapped confidence interval for metrics returned by
           function 'metrics_from_binvecs',  given the input threshold.
       ---------------------------------------------------------------------------------
       Args:
           * y_true_bin
               - flat numpy array
           * y_hat
               - numpy array, predictions by the model, should contain probability values within range [0,1]
           * thresh
               - a scalar, the cut-off threshold to binarize the predictions in y_hat, within range [0,1]
           * nsample
               - number of boostrapped samples to generate
           * q
               - the quantile (percentile) for the confidence interval (ci). The ci is defined as (1, 1-q).
               For the detailed method to determine the boundaries, see the implementation below (bootstrap_metrics.quantile)
       ---------------------------------------------------------------------------------
       Returns:
           * A 3-row dataframe containing the mid value and ci for each metric
               index: ci quantiles and 'mid'
               columns: metrics

       ---------------------------------------------------------------------------------
    """
    assert 0.0 <= thresh <= 1.0

    print(f"Bootstrapping {nsample} samples, interval=({q},{1-q})")
    bootstrap_metrics = {}
    y_hat_bin = np.where((y_hat >= thresh), 1, np.where((y_hat < thresh), 0, np.nan))

    for i in tqdm.tqdm(range(nsample)):
        resampled_idx = np.random.choice(y_true_bin.size, y_hat_bin.size)
        bootstrap_metrics[i] = metrics_from_binvecs(
            y_true_bin[resampled_idx], y_hat_bin[resampled_idx]
        )
        # calculate ROC-AUC
        bootstrap_metrics[i]["roc_auc"] = compute_roc_auc(
            y_true_bin[resampled_idx], y_hat[resampled_idx]
        )

    bootstrap_metrics = pd.DataFrame.from_dict(bootstrap_metrics, orient="index")
    ci = bootstrap_metrics.quantile(q=[q, 1 - q], interpolation="nearest")

    mid_vals_dict = metrics_from_binvecs(y_true_bin, y_hat_bin)
    mid_vals_dict["roc_auc"] = compute_roc_auc(y_true_bin, y_hat)
    mid_vals = pd.DataFrame.from_dict({"mid": mid_vals_dict}, orient="index")

    mid_cis = pd.concat([mid_vals, ci], axis=0)
    mid_cis["n_bootstrap"] = nsample
    mid_cis["thresh"] = thresh
    return mid_cis


def plot_preds_distribution(y_true, y_hat, phase="", epoch=""):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        *  Visualize prediction (y_hat) distributions as violin plot
    ---------------------------------------------------------------------------------
    Args:
        * y_true
            - 1D numpy array, should contain only binary values (0,1 or bool)
        * y_hat
            - numpy array, predictions by the model, should contain probability values within range [0,1]
        * phase
            - string
        * epoch
            - string
    ---------------------------------------------------------------------------------
    """

    ytrue_yhat = pd.DataFrame(data=y_true, columns=["ytrue"])
    ytrue_yhat["yhat"] = y_hat

    # display(ytrue_yhat.head(n=2))
    # display(ytrue_yhat.yhat.dtype)
    # display(ytrue_yhat.ytrue.dtype)
    # display(phase)

    g = sns.catplot(x="ytrue", y="yhat", data=ytrue_yhat, kind="violin")
    g.set(ylim=(-0.2, 1.2))
    g.fig.suptitle(f"{phase} pred distribution, epoch{epoch}")


def plot_moving_thresh_metrics(metrics_moving_thresh, optim_thresh=None, phase="", epoch=""):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        *  Plot metrics as a function of the moving threshold, with the optimal threshold.
    ---------------------------------------------------------------------------------
    Args:
        * metrics_moving_thresh
            - A dataframe containing the metrics
                    index: cut-off threshold
                    columns: metrics probabilities
        * optim_thresh
            - a scalar
        * phase
            - string
        * epoch
            - string
    ---------------------------------------------------------------------------------
    """
    ax = metrics_moving_thresh[
        ["f1_score", "g_mean", "youdens_index", "sensitivity", "ppv"]
    ].plot()
    if optim_thresh:
        ax.vlines(optim_thresh, 0, 1, colors="k")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid("on", linestyle="dashed")
    ax.set_title(f"{phase} metrics moving thresh, epoch{epoch}")


def bootstrap_multicalss_metrics(y_true, y_hat, num_classes, nsample=1000, q=0.05):
    """
     ---------------------------------------------------------------------------------
        Purpose:
            * Compute bootstrapped confidence interval for metrics returned by
            function 'metrics_from_binvecs',  given the input threshold.
    ---------------------------------------------------------------------------------
    Args:
        * y_true
            - numpy array of shape n x c, n- number of samples, c- number of classes, one-hot-encoded
        * y_hat
            - numpy array, predictions by the model, of shape n x c, n- number of samples, c- number of classes, elements are probabilities
        * nsample
            - number of boostrapped samples to generate
        * q
            - the quantile (percentile) for the confidence interval (ci). The ci is defined as (1, 1-q).
            For the detailed method to determine the boundaries, see the implementation below (bootstrap_metrics.quantile)
    ---------------------------------------------------------------------------------
    Returns:
        * A 3-row dataframe containing the mid value and ci for each metric
            index: ci quantiles and 'mid'
            columns: metrics
    ---------------------------------------------------------------------------------
    """
    print(f"Bootstrapping {nsample} samples, interval=({q},{1-q})")
    bootstrap_metrics = {}
    for i in tqdm.tqdm(range(nsample)):
        resampled_idx = np.random.choice(y_true.shape[0], y_hat.shape[0])
        bootstrap_metrics[i] = compute_multiclass_metrics(
            y_true[resampled_idx, :], y_hat[resampled_idx, :], num_classes
        ).to_dict()
        bootstrap_metrics[i].update(
            compute_roc_auc(y_true[resampled_idx, :], y_hat[resampled_idx, :])
        )

    bootstrap_metrics = pd.DataFrame.from_dict(bootstrap_metrics, orient="index")
    ci = bootstrap_metrics.quantile(q=[q, 1 - q], interpolation="nearest")

    mid_auc = pd.DataFrame.from_dict({"mid": compute_roc_auc(y_true, y_hat)}, orient="index")
    mid_vals = pd.DataFrame.from_dict(
        {"mid": compute_multiclass_metrics(y_true, y_hat, num_classes)}, orient="index"
    )
    mid_vals = pd.concat([mid_vals, mid_auc], axis=1)
    mid_cis = pd.concat([mid_vals, ci], axis=0)
    mid_cis["n_bootstrap"] = nsample
    return mid_cis


def display_examples_from_model(
    model, dataloader, device, directory="display_video000/", N_POOL=300, resize_dims=(256, 256)
):
    """
    Display examples by saving videos and predictions to a specified directory.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the model on.
        directory (str, optional): Directory where the videos will be saved.
        N_POOL (int, optional): Number of examples to display.
        resize_dims (tuple, optional): Dimensions to resize the videos.

    Creates video files and a CSV with predictions and targets.
    """
    csv_path = os.path.join(directory, "pred_display.csv")
    dict_video = {"data": [], "pred": [], "target": []}
    counter = 0

    if not os.path.exists(directory):
        os.makedirs(directory)

    model.eval()
    with torch.no_grad():
        for X, targets, _ in tqdm(dataloader, desc="Displaying Examples"):
            X = X.to(device)
            targets = targets.to(device)
            outputs = model(X)  # Assuming the model returns the required outputs

            for j in range(X.shape[0]):
                if counter < N_POOL:
                    video_path = os.path.join(directory, f"{counter}.mp4")
                    counter += 1

                    pred = outputs[j].detach().cpu().numpy()
                    target = targets[j].item()

                    dict_video["data"].append(video_path)
                    dict_video["pred"].append(pred)
                    dict_video["target"].append(target)

                    temp = Resize(resize_dims)(X[j]).cpu().numpy()
                    temp = (temp - temp.min()) * 255 / (temp.max() - temp.min())
                    imageio.mimwrite(
                        video_path, np.transpose(temp.astype(np.uint8), (1, 2, 3, 0)), fps=15
                    )

            if counter >= N_POOL:
                break

    pd.DataFrame(dict_video).to_csv(csv_path)
    print(f"Saved example videos and predictions to {directory}")


# Example usage
# Assuming you have a trained model and a dataloader
# display_examples(trained_model, your_dataloader, torch.device('cuda'))


def display_examples(data, directory="display_video000/", N_POOL=300, resize_dims=(256, 256)):
    """
    Display examples by saving videos and predictions to a specified directory.

    Args:
        data (list): A list of tuples, each containing (X, outputs, targets), where
                     X is the video data, outputs are the model predictions, and
                     targets are the ground truth labels.
        directory (str, optional): Directory where the videos will be saved.
        N_POOL (int, optional): Number of examples to display.
        resize_dims (tuple, optional): Dimensions to resize the videos.

    Creates video files and a CSV with predictions and targets.
    """
    csv_path = os.path.join(directory, "pred_display.csv")
    dict_video = {"data": [], "pred": [], "target": []}
    counter = 0

    if not os.path.exists(directory):
        os.makedirs(directory)

    for X, outputs, targets in tqdm(data, desc="Displaying Examples"):
        for j in range(X.shape[0]):
            if counter < N_POOL:
                video_path = os.path.join(directory, f"{counter}.mp4")
                counter += 1

                pred = outputs[j].detach().cpu().numpy()
                target = targets[j].item()

                dict_video["data"].append(video_path)
                dict_video["pred"].append(pred)
                dict_video["target"].append(target)

                temp = Resize(resize_dims)(X[j]).cpu().numpy()
                temp = (temp - temp.min()) * 255 / (temp.max() - temp.min())
                imageio.mimwrite(
                    video_path, np.transpose(temp.astype(np.uint8), (1, 2, 3, 0)), fps=15
                )

            if counter >= N_POOL:
                break

    pd.DataFrame(dict_video).to_csv(csv_path)
    print(f"Saved example videos and predictions to {directory}")


def dice_similarity_coefficient(inter, union):
    """
    ---------------------------------------------------------------------------------
    Purpose:
        * Computes the dice similarity coefficient
    ---------------------------------------------------------------------------------
    Args:
        * inter:
            - iterable: iterable of the intersections
        * union:
            - iterable: iterable of the unions
    ---------------------------------------------------------------------------------
    Returns:
        * dice similarity coefficient
    ---------------------------------------------------------------------------------
    Sanity Checks / Bug fixes:
        * None
    ---------------------------------------------------------------------------------
    Called By:
        * Files
            orion.utils.video_class regression
    ---------------------------------------------------------------------------------
    TODO:
        * None
    """
    return 2 * sum(inter) / (sum(union) + sum(inter))


def initialize_regression_metrics(device):
    metrics = torchmetrics.MetricCollection(
        {
            "mae": torchmetrics.MeanAbsoluteError().to(device),
            "mse": torchmetrics.MeanSquaredError().to(device),
        }
    )
    return metrics


def log_regression_metrics_to_wandb(phase, metrics_dict, loss, learning_rate=None):
    # Log each metric in the metrics_dict
    for metric_name, metric_tensor in metrics_dict.items():
        # Convert tensor to a standard Python number (like float)
        metric_value = metric_tensor.item() if torch.is_tensor(metric_tensor) else metric_tensor
        # print(f"{metric_name}: {metric_value}")
        wandb.log({f"{phase}_{metric_name}": metric_value})

    # Log the loss
    loss_value = loss.item() if torch.is_tensor(loss) else loss
    wandb.log({f"{phase}_loss": loss_value})

    # Optionally log the learning rate
    if learning_rate is not None:
        lr_value = learning_rate.item() if torch.is_tensor(learning_rate) else learning_rate
        wandb.log({"learning_rate": lr_value})


def update_best_regression_metrics(final_metrics, best_metrics):
    # Extract metric values, converting from tensor if necessary
    mae = (
        final_metrics["mae"].item()
        if torch.is_tensor(final_metrics["mae"])
        else final_metrics["mae"]
    )
    mse = (
        final_metrics["mse"].item()
        if torch.is_tensor(final_metrics["mse"])
        else final_metrics["mse"]
    )

    mse_tensor = torch.tensor(mse) if not isinstance(mse, torch.Tensor) else mse
    rmse = torch.sqrt(mse_tensor)

    # Update MAE if it's better or if best_mae is not set
    if best_metrics["best_mae"] is None or mae < best_metrics["best_mae"]:
        best_metrics["best_mae"] = mae

    # Update RMSE if it's better or if best_rmse is not set
    if best_metrics["best_rmse"] is None or rmse < best_metrics["best_rmse"]:
        best_metrics["best_rmse"] = rmse

    return best_metrics, mae, mse, rmse


def log_binary_classification_metrics_to_wandb(
    phase,
    loss,
    auc_score,
    optimal_threshold,
    y_true,
    pred_labels,
    label_map,
    learning_rate=None,
):
    # Log binary classification metrics to WandB
    wandb.log({f"{phase}_epoch_loss": loss})
    wandb.log({f"{phase}_auc": auc_score})
    wandb.log({f"{phase}_optimal_thresh": optimal_threshold})
    # Convert predictions to label format

    if pred_labels.ndim > 1 and pred_labels.shape[1] == 1:
        pred_labels = pred_labels.flatten()

    # Define your class names (replace with actual class names)
    # Check if label_map is defined, if not, generate class names based on unique classes in y_true
    if label_map is None:
        class_names = ["0", "1"]
    else:
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
    head_name,
    loss,
    y_true,
    predictions,
    learning_rate=None,
):
    """Log multi-class metrics to wandb.

    Args:
        phase (str): The phase of the evaluation (e.g., "train", "val").
        epoch (int): The epoch number.
        metrics_summary (dict): A dictionary containing the computed metrics summary.
        labels_map (dict): A dictionary mapping class indices to class labels.
        y_true (numpy.ndarray): The true labels.
        predictions (numpy.ndarray): The predicted probabilities.
        learning_rate (float, optional): The current learning rate.
    """
    # Retrieve metrics
    roc_auc = metrics_summary["auc"]
    conf_mat = metrics_summary["confmat"]

    # Create default labels if labels_map is None
    if labels_map is None:
        num_classes = conf_mat.shape[0]  # Get number of classes from confusion matrix
        labels_map = {f"Class {i}": i for i in range(num_classes)}

    # Log individual class AUCs
    for label in labels_map:
        wandb.log(
            {
                f"{phase}_roc_auc_macro_for_head_{head_name}_class_{label}": roc_auc[
                    labels_map[label]
                ]
            }
        )

    if learning_rate is not None:
        wandb.log({"learning_rate": learning_rate})

    # For micro averaged AUC
    wandb.log({f"{phase}_roc_auc_micro_for_head_{head_name}": metrics_summary["auc_weighted"]})

    # Log the loss
    wandb.log({f"{phase}_loss_for_head_{head_name}": loss})

    # Log the confusion matrix as an image
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert labels_map values to list for xticklabels and yticklabels
    labels_list = list(labels_map.keys())

    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_list,
        yticklabels=labels_list,
        ax=ax,
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"{phase.capitalize()} Confusion Matrix - Epoch {epoch}")

    # Log the confusion matrix as an image
    wandb.log({f"{phase}_epoch_{epoch}_confusion_matrix": wandb.Image(fig)})
    log_multiclass_roc_curve_to_wandb(
        y_true, predictions, labels_map, roc_auc, phase, epoch, head_name
    )


def log_multiclass_roc_curve_to_wandb(
    y_true, y_pred, labels_map, auc_array, phase, epoch, head_name
):
    """
    Log multi-class ROC curves to wandb.

    Args:
        y_true (torch.Tensor or numpy.ndarray): True class labels (integer encoded).
        y_pred (numpy.ndarray): Predicted probabilities for each class.
        labels_map (dict): Mapping from class indices to class names.
        auc_array (numpy.ndarray): Array of AUC values for each class.
        phase (str): Current phase ('train', 'val', etc.).
        epoch (int): Current epoch.
    """
    # Ensure y_true is of integer type
    y_true = y_true.astype(int)

    # Convert y_true to one-hot encoding if it's not already
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        y_true = torch.nn.functional.one_hot(
            torch.tensor(y_true), num_classes=len(labels_map)
        ).numpy()

    num_classes = y_true.shape[1]
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

    labels_list = list(labels_map.keys())
    for i, color in zip(range(num_classes), colors):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        label = labels_list[i]
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"ROC curve for {label} (area = {auc_array[i]:0.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve for head {head_name} - {phase.capitalize()} - Epoch {epoch}")
    ax.legend(loc="lower right")

    wandb.log({f"{phase}_roc_curve_for_head_{head_name}": wandb.Image(fig)})


def compute_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_index = tpr - fpr
    optimal_idx = np.argmax(j_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


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
    # print(list(metrics.keys()))

    if num_classes <= 2:
        # Update for Binary Classification
        metrics["auc"].update(preds, target.int())
        if preds.ndim > 1 and preds.shape[1] == 2:  # Check if preds have two columns
            preds = preds[:, 1]  # Use the second column for binary classification
        elif preds.ndim > 1 and preds.shape[1] == 1:  # Check if preds have one column
            preds = preds.squeeze()  # Squeeze the dimension
        metrics["confmat"].update((preds > 0.5).int(), target.int())
    else:
        # Update for Multi-Class Classification
        # Convert predictions to label format
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.flatten()
        # Ensure target is an integer tensor
        target = target.long()
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes)
        metrics["auc"].update(preds, target_one_hot)
        metrics["auc_weighted"].update(preds, target)
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
        try:
            # Compute the metric value
            computed_value = metric.compute()

            # Check if tensor has a single element
            if computed_value.numel() == 1:
                computed_metrics[metric_name] = computed_value.item()
            else:
                # Convert to a floating point tensor before taking the mean
                computed_value_float = computed_value.type(torch.float32)
                computed_metrics[metric_name] = computed_value_float.mean().item()
        except Exception as e:
            print(f"Error computing metric {metric_name}: {e}")
            # Add more detailed debugging information
            print(f"Metric state: {metric.state_dict()}")
            raise e

    # Reset the metrics after computation
    for metric in metrics.values():
        metric.reset()

    return computed_metrics
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
        try:
            computed_value = metric.compute()

            # Check if tensor has a single element
            if computed_value.numel() == 1:
                computed_metrics[metric_name] = computed_value.item()
            else:
                # Convert to a floating point tensor before taking the mean
                computed_value_float = computed_value.type(torch.float32)
                computed_metrics[metric_name] = computed_value_float.mean().item()
        except Exception as e:
            print(f"Error computing metric {metric_name}: {e}")
            raise e

    for metric in metrics.values():
        metric.reset()

    return computed_metrics


def compute_multiclass_metrics(metrics):
    """
    Computes multi-class classification metrics based on the provided metrics dictionary.

    Args:
        metrics (dict): A dictionary containing the metrics to compute.

    Returns:
        computed_metrics (dict): A dictionary containing the computed classification metrics.

    Examples:
        >>> metrics = {'auc': AUROC(), 'auc_micro': AUROC(), 'confmat': ConfusionMatrix()}
        >>> computed_metrics = compute_multiclass_metrics(metrics)
    """

    computed_metrics = {}
    for metric_name, metric in metrics.items():
        computed_value = metric.compute()
        # Handling different types of metric outputs
        if isinstance(computed_value, torch.Tensor):
            if computed_value.numel() == 1:
                # Single value metrics (e.g., AUC, AUC Micro)
                computed_metrics[metric_name] = computed_value.item()
            else:
                # Multi-value metrics (e.g., Confusion Matrix)
                computed_metrics[metric_name] = computed_value.cpu().numpy()
        else:
            # In case of non-tensor metrics (if any)
            computed_metrics[metric_name] = computed_value

    for metric in metrics.values():
        metric.reset()

    return computed_metrics


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
            "auc": torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average=None).to(
                device
            ),
            "auc_weighted": torchmetrics.AUROC(
                task="multiclass", num_classes=num_classes, average="weighted"
            ).to(device),
            "confmat": torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=num_classes
            ).to(device),
        }
    return metrics
