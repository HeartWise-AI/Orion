import torch
import torch.nn as nn

from orion.models.pytorchvideo_models import get_pretrained_model


# Define the model
class StochasticRegressor(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 128)  # assuming input features are of size 10
        self.fc_mu = nn.Linear(128, 1)  # output layer for predicting mean
        self.fc_sigma = nn.Linear(128, 1)  # output layer for predicting standard deviation

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))  # ensure sigma is positive
        return mu, sigma


def get_dist_model(model_name):
    model = get_pretrained_model(model_name, num_class=1, model_type="regress")

    if "swin3d" in model_name:
        model = model[0]
        n_inputs = model.head.in_features

        model2 = StochasticRegressor(n_inputs)
        model.head = model2
    return model
