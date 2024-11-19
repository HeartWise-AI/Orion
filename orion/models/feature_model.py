import torch
import torch.nn as nn
from oread.models.pytorchvideo_models import get_pretrained_model


# +
# Define the model
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Regressor(nn.Module):
    def __init__(self, model_name, num_class=1, num_features=12):
        super().__init__()
        model = get_pretrained_model(model_name, num_class=num_class, task="regression")
        model = model[0]
        n_inputs = model.head.in_features
        # print('n_inputs', n_inputs)
        model.head = Identity()

        self.model = model
        self.fc = nn.Linear(n_inputs + num_features, num_class)

    def forward(self, x):
        features = self.model(x[0])

        age = x[1] / 120
        segment = torch.nn.functional.one_hot(x[2].squeeze(), num_classes=11).to(torch.float32)

        x = torch.cat([features, segment, age], 1)

        x = self.fc(x)
        return x


# -


def get_fmodel(model_name):
    if "swin3d" in model_name:
        model = Regressor(model_name)
    return model


# +
# model = get_dist_model('swin3d_b')

# +
# x = [torch.zeros(2,3,24,224,224), torch.zeros(2,1), torch.zeros(2,1)]
# model(x)
# -
