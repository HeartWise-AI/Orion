import torch
from torch.utils.data import DataLoader
from data_loader import VideoDataset
from models import load_model
from config import Config
from utils import load_checkpoint, save_metrics
from tqdm import tqdm

if __name__ == "__main__":
    config = Config("config.yaml", "TEST")

    # Load test dataset
    test_dataset = VideoDataset(config, "TEST")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    # Load the model
    model = load_model(config).cuda()
    
    # Load model checkpoint
    model = load_checkpoint(model, config.model_path)

    model.cuda()

    """Make predictions with a trained model."""
    model.eval()
    predictions_list = []
    labels_list = []
    exams_list = []
    filenames_list = []
    with torch.no_grad():
        for inputs, labels, exams, filenames in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.cuda().float()
            outputs = model(inputs)
            predictions_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.numpy())
            exams_list.extend(exams)
            filenames_list.extend(filenames)

    save_metrics(labels_list, predictions_list, exams_list, filenames_list, config)

    print(f"Predictions and Corresponding Labels have been logged.")
