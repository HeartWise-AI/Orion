# train.py
import torch
import numpy as np
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm
import os
import random
import string

from utils import log_metrics

def train_model(model, train_loader, val_loader, config):
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        total = 0
        train_all_outputs = []
        train_all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=False)

        for inputs, labels in pbar:
            inputs, labels = inputs.cuda().float(), labels.cuda().float()
            config.optimizer.zero_grad()
            with torch.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = config.criterion(outputs.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            config.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            mean_absolute_error = MeanAbsoluteError().to('cuda')
            mae = mean_absolute_error(outputs.squeeze(), labels)
            running_mae += mae.item() * inputs.size(0)
            total += labels.size(0)

            train_all_outputs.append(outputs.detach().cpu().numpy().squeeze())
            train_all_labels.append(labels.detach().cpu().numpy().squeeze())
            pbar.set_postfix({"loss": running_loss / total, "mae": running_mae / total})

        epoch_loss = running_loss / total
        epoch_mae = running_mae / total
        train_all_outputs = np.concatenate(train_all_outputs)
        train_all_labels = np.concatenate(train_all_labels)

        print(f'Epoch {epoch}/{config.num_epochs - 1}, Loss: {epoch_loss:.4f}, mae: {epoch_mae:.4f}')
        val_loss, val_mae, val_all_outputs, val_all_labels = validate_model(model, val_loader, config.criterion)
        
        if config.use_wandb:
            log_metrics(epoch, epoch_loss, epoch_mae, val_loss, val_mae, train_all_labels, train_all_outputs, val_all_labels, val_all_outputs, config.optimizer, threshold=config.threshold)

        config.scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save the model
            torch.save(model.state_dict(), f"{config.model_path}/best_model.pth")
            print(f"Model saved at epoch {epoch} with validation loss {val_loss:.4f}")

def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda().float(), labels.cuda().float()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            running_loss += loss.item() * inputs.size(0)
            mean_absolute_error = MeanAbsoluteError().to('cuda')
            mae = mean_absolute_error(outputs.squeeze(), labels)
            running_mae += mae.item() * inputs.size(0)
            total += labels.size(0)
            all_outputs.append(outputs.detach().cpu().numpy().squeeze())
            all_labels.append(labels.detach().cpu().numpy().squeeze())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    val_loss = running_loss / total
    val_mae = running_mae / total

    return val_loss, val_mae, all_outputs, all_labels

