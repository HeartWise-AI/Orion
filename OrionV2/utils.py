# utils.py
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import av
import cv2 as cv
from sklearn.metrics import auc, confusion_matrix, roc_curve, roc_auc_score
import torch
import wandb
import pandas as pd

def read_with_pyav(video_path, model_name, frames_size):
    container = av.open(video_path)
    frames = []
    if (model_name == "x3d_xs" or model_name == "x3d_s"):
        resizeShape = (frames_size, frames_size)
    elif model_name == "x3d_m":
        resizeShape = (frames_size, frames_size)
    
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        img = cv.resize(img, resizeShape)
        minImg = np.min(img)
        maxImg = np.max(img)
        img = (img - minImg) / (maxImg - minImg)
        frames.append(img)
    
    return np.array(frames)

def select_random_frames_with_padding(array, num_frames=32):
    X = array.shape[0]
    if X >= num_frames:
        start_index = np.random.randint(0, X - num_frames + 1)
        selected_frames = array[start_index : start_index + num_frames, :, :, :]
    else:
        selected_frames = array
        last_frame = array[-1, :, :, :]
        padding = np.zeros((num_frames - X, *array.shape[1:]), dtype=array.dtype)
        padding[:] = last_frame
        selected_frames = np.concatenate((selected_frames, padding), axis=0)
    
    return selected_frames


def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pth')))
    return model


def log_metrics(epoch, epoch_loss, epoch_mae, val_loss, val_mae, train_labels, train_outputs, val_labels, val_outputs, optimizer, threshold=0.5):
    data_train = [[x, y] for (x, y) in zip(train_labels, train_outputs)]
    tableTrain = wandb.Table(data=data_train, columns = ["Labels", "Predictions"])

    data_valid = [[x, y] for (x, y) in zip(val_labels, val_outputs)]
    tableValid = wandb.Table(data=data_valid, columns = ["Labels", "Predictions"])

    binary_train_labels = np.where(train_labels > threshold, 1, 0)
    binary_train_outputs = np.where(train_outputs > threshold, 1, 0)
    conf_matrix_train = confusion_matrix(binary_train_labels, binary_train_outputs)

    binary_val_labels = np.where(val_labels > threshold, 1, 0)
    binary_val_outputs = np.where(val_outputs > threshold, 1, 0)
    conf_matrix_val = confusion_matrix(binary_val_labels, binary_val_outputs)

    figTrain, axTrain = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"], ax=axTrain)
    axTrain.set_xlabel("Predicted Labels")
    axTrain.set_ylabel("True Labels")
    axTrain.set_title(f"Training Confusion Matrix - Epoch {epoch}")

    figValid, axValid = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix_val, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"], ax=axValid)
    axValid.set_xlabel("Predicted Labels")
    axValid.set_ylabel("True Labels")
    axValid.set_title(f"Validation Confusion Matrix - Epoch {epoch}")

    fpr, tpr, _ = roc_curve(binary_train_labels, train_outputs)
    auc_train = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(binary_val_labels, val_outputs)
    auc_valid = auc(fpr, tpr)

    wandb.log({
        "epoch": epoch,
        "train_loss": epoch_loss,
        "train_mae": epoch_mae,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "roc_auc_train": auc_train,
        "roc_auc_valid": auc_valid,
        f"training_correlation_plot_{epoch}" : wandb.plot.scatter(tableTrain, "Labels", "Predictions", title=f"Training Correlation {epoch}"),
        f"validation_correlation_plot_{epoch}" : wandb.plot.scatter(tableValid, "Labels", "Predictions", title=f"Validation Correlation {epoch}"),
        f"train_confusion_matrix_{epoch}": wandb.Image(figTrain),
        f"validation_confusion_matrix_{epoch}": wandb.Image(figValid)
    })


    plt.close(figTrain)
    plt.close(figValid)


def save_metrics(labels_list, predictions_list, exams_list, filenames_list, config):
    predictions = np.array(predictions_list).squeeze()
    labels = np.array(labels_list)
    suffix = f"stride_{config.stride}_predictions_on_dataset_{config.dataset_origin}"

    # Get metrics for exam level
    df = pd.DataFrame({"exams": exams_list, "labels": labels, "predictions": predictions, config.filename_column: filenames_list})   

    # Save predictions to csv
    df.to_csv(os.path.join(config.model_path, f"predictions_{suffix}.csv"), index=False)

    df_grouped = df.groupby(['exams']).mean(numeric_only=True)
    ## LONGER VIDEOS HAVE MORE WEIGHT IN THE GROUPBY, SHOULD WE ADDRESS THIS?

    labels_binary = np.where(labels > config.threshold, 1, 0)
    predictions_binary = np.where(predictions > config.threshold, 1, 0)
    labels_binary_grouped = np.where(df_grouped['labels'] > config.threshold, 1, 0)
    predictions_grouped = np.where(df_grouped['predictions'] > config.threshold, 1, 0)

    conf_matrix = confusion_matrix(labels_binary, predictions_binary)
    conf_matrix_grouped = confusion_matrix(labels_binary_grouped, predictions_grouped)
    
    # Save Confusion Matrix (Video level)
    figCM, axCM = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"], ax=axCM)
    axCM.set_xlabel("Predicted Labels")
    axCM.set_ylabel("True Labels")
    axCM.set_title(f"Test Confusion Matrix - {config.dataset_origin} - Video Level")
    plt.savefig(os.path.join(config.model_path, f"confusion_matrix_video_{suffix}.png"))
    plt.close(figCM)

    # Save Confusion Matrix (Study level)
    figCM, axCM = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix_grouped, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"], ax=axCM)
    axCM.set_xlabel("Predicted Labels")
    axCM.set_ylabel("True Labels")
    axCM.set_title(f"Test Confusion Matrix - {config.dataset_origin} - Study Level")
    plt.savefig(os.path.join(config.model_path, f"confusion_matrix_study_{suffix}.png"))
    plt.close(figCM)

    # Save ROC Curve
    fpr_grouped, tpr_grouped, _ = roc_curve(labels_binary_grouped, df_grouped['predictions'].values)
    auc_test_grouped = auc(fpr_grouped, tpr_grouped)
    fpr, tpr, _ = roc_curve(labels_binary, predictions)
    auc_test = auc(fpr, tpr)

    figROC, axROC = plt.subplots(figsize=(10, 8))
    axROC.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve video level (area = {auc_test:.2f})')
    axROC.plot(fpr_grouped, tpr_grouped, color='darkblue', lw=2, label=f'ROC curve study level (area = {auc_test_grouped:.2f})')
    axROC.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axROC.set_xlim([0.0, 1.0])
    axROC.set_ylim([0.0, 1.05])
    axROC.set_xlabel('False Positive Rate')
    axROC.set_ylabel('True Positive Rate')
    axROC.set_title(f'Receiver Operating Characteristic - {config.dataset_origin} - Stride - {config.stride}')
    axROC.legend(loc="lower right")
    plt.savefig(os.path.join(config.model_path, f"roc_curve_{suffix}.png"))
    plt.close(figROC)
