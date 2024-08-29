# main.py
import wandb
import torch
from torch.utils.data import DataLoader, random_split

from data_loader import VideoDataset, DumbVideoDataset
from models import load_model
from training_loop import train_model
from config import Config

def runTestCase():
    config = Config("config.yaml")
    config.model_name = "x3d_xs"
    config.pretrained = True
    config.problem_type = "regression"
    config.classes = 1
    config.optimizer_name = "Adam"
    config.num_epochs = 10
    
    dataset = DumbVideoDataset()

    # Load and modify the pre-trained model based on config
    model = load_model(config.model_name, config.pretrained, config.problem_type, config.classes)
    config._initialize_optimizer(model)
    config._initialize_scheduler()

    # Prepare Dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = model.cuda()

    # Train and Validate Model
    model = train_model(
        model, 
        train_loader, 
        val_loader,
        config
    )

    if config.use_wandb:
        wandb.finish()
