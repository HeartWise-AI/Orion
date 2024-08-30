# main.py
import os
import random
import string
import wandb
from torch.utils.data import DataLoader
from data_loader import VideoDataset, DumbVideoDataset
from models import load_model
from training_loop import train_model
from config import Config

if __name__ == "__main__":
    config = Config("config.yaml", split="TRAIN")
    directory_name = f"{config.model_name}_{config.dataset_origin}_{config.fps}_{config.optimizer_name}_{config.learning_rate}_{config.criterion_name}_{config.problem_type}_{config.pretrained}_{config.num_frames}_{config.batch_size}"

    # Handle the special "TESTCASE" mode (black and white video classification)
    if config.dataset_origin == "TESTCASE":
        dataset = DumbVideoDataset()
        exit()

    # Initialize the directory for the model            
    while True:
        directory = f"experiments/{directory_name}_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        if not os.path.exists(directory):
            os.makedirs(directory)
            break

    # Initialize wandb conditionally
    if config.use_wandb:
        wandb.init(project="X3D Test", config=config)
        wandb.run.name = directory.replace("experiments/", "")
    
    # Load Dataset
    train_dataset = VideoDataset(config, "TRAIN")
    val_dataset = VideoDataset(config, "VAL")

    # Load and modify the pre-trained model based on config
    model = load_model(config)
    config._initialize_model_path(directory)
    config._initialize_optimizer(model)
    config._initialize_scheduler()
    config.save(os.path.join(directory, 'config.yaml'))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = model.cuda()

    # Train and Validate Model
    train_model(
        model, 
        train_loader, 
        val_loader,
        config
    )

    if config.use_wandb:
        wandb.finish()
