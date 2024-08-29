# models.py
import torch.nn as nn
import torch

def load_model(config):
    # Load the specified pre-trained X3D model
    model = torch.hub.load('facebookresearch/pytorchvideo', config.model_name, pretrained=config.pretrained)

    # Replace the last layer depending on the problem type
    if config.problem_type == "regression":
        model.blocks[5].proj = nn.Linear(in_features=2048, out_features=1, bias=True)
    elif config.problem_type == "classification":
        model.blocks[5].proj = nn.Linear(in_features=2048, out_features=config.classes, bias=True)
    
    return model
