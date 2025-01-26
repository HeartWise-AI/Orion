import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from orion.models.contrastive_model import ContrastiveModel


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch}") as pbar:
        for batch in dataloader:
            videos, texts = batch
            videos = videos.to(device)

            # Forward pass
            outputs = model(videos, texts)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})

            # Log to wandb
            wandb.log({"train_loss": loss.item(), "epoch": epoch})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            videos, texts = batch
            videos = videos.to(device)

            # Forward pass
            outputs = model(videos, texts)
            loss = outputs["loss"]
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_contrastive_model(
    train_dataloader, val_dataloader, config, device, num_epochs=60, learning_rate=1e-4
):
    # Initialize model
    model = ContrastiveModel(
        temperature=config.get("temperature", 0.07), output_dim=config.get("output_dim", 512)
    ).to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.get("weight_decay", 0.01)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Initialize wandb
    wandb.init(
        project="orion-contrastive",
        config={
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "temperature": config.get("temperature", 0.07),
            "output_dim": config.get("output_dim", 512),
            "weight_decay": config.get("weight_decay", 0.01),
            "architecture": "mvit-pubmedbert",
        },
    )

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_dataloader, optimizer, device, epoch)

        # Validation
        val_loss = validate(model, val_dataloader, device)

        # Update learning rate
        scheduler.step()

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                "best_contrastive_model.pth",
            )

    wandb.finish()
    return model


def load_contrastive_model(model_path, config, device):
    model = ContrastiveModel(
        temperature=config.get("temperature", 0.07), output_dim=config.get("output_dim", 512)
    ).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
