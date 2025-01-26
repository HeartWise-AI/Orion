"""
Defining models used for orion
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from stam_pytorch import STAM
from timesformer_pytorch import TimeSformer

from orion.models.config import _C
from orion.models.videopairclassifier import VideoPairClassifier
from orion.models.vivit import ViViT
from orion.models.x3d_multi import Bottleneck as MultiBottleneck
from orion.models.x3d_multi import X3D_multi


def get_inplanes(version):
    if version == "S":
        return [(24, 24), (48, 48), (96, 96), (192, 192)]
    elif version == "M":
        return [(24, 24), (48, 48), (96, 96), (192, 192)]
    elif version == "L":
        return [(32, 32), (64, 64), (128, 128), (256, 256)]
    else:
        raise ValueError(f"Unsupported X3D version: {version}")


def get_blocks(version):
    if version == "S":
        return [3, 5, 11, 7]
    elif version == "M":
        return [3, 5, 11, 7]
    elif version == "L":
        return [5, 10, 25, 15]
    else:
        raise ValueError(f"Unsupported X3D version: {version}")


def x3d_multi(num_classes, **kwargs):
    model = X3D_multi(
        MultiBottleneck, get_blocks("M"), get_inplanes("M"), num_classes=num_classes, **kwargs
    )
    return model


def x3d_legacy(num_classes, **kwargs):
    model = X3D_legacy(
        LegacyBottleneck, get_blocks("M"), get_inplanes("M"), n_classes=num_classes, **kwargs
    )
    return model


def timesformer(num_classes, resize, **kwargs):
    """Constructs a timesformer model."""
    model = TimeSformer(
        dim=512,
        image_size=resize,
        patch_size=16,
        num_frames=64,
        num_classes=num_classes,
        depth=12,
        heads=8,
        dim_head=64,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )
    return model


def vivit(num_classes, resize, num_frames, **kwargs):
    """Constructs a timesformer model."""
    model = ViViT(resize, num_classes, num_frames)
    return model


def movinet(num_classes, resize, **kwargs):
    """Constructs a timesformer model."""
    model = MoViNet(_C.MODEL.MoViNetA3, num_classes=num_classes)
    return model


def stam(num_classes, resize, **kwargs):
    model = STAM(
        dim=512,
        image_size=resize,  # size of image
        patch_size=32,  # patch size
        num_frames=64,  # number of image frames, selected out of video
        space_depth=12,  # depth of vision transformer
        space_heads=8,  # heads of vision transformer
        space_mlp_dim=2048,  # feedforward hidden dimension of vision transformer
        time_depth=6,  # depth of time transformer (in paper, it was shallower, 6)
        time_heads=8,  # heads of time transformer
        time_mlp_dim=2048,  # feedforward hidden dimension of time transformer
        num_classes=num_classes,  # number of output classes
        space_dim_head=64,  # space transformer head dimension
        time_dim_head=64,  # time transformer head dimension
        dropout=0.1,  # dropout
        emb_dropout=0.1,
    )  # embedding dropout
    return model


# New functions for loading and modifying models


class RegressionHead(nn.Module):
    def __init__(self, dim_in, num_classes=1):
        super().__init__()
        self.fc1 = nn.Conv3d(dim_in, 2048, bias=True, kernel_size=1, stride=1)
        self.regress = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = x.mean([2, 3, 4])
        x = self.regress(x)
        return x


class TransformerHead(nn.Module):
    def __init__(self, dim_in, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        # Global average pooling over the spatial and temporal dimensions
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MultiOutputHead(nn.Module):
    def __init__(self, dim_in, head_structure: dict[str, int]):
        """
        Initialize a multi-output head with different number of classes per head.

        Args:
            dim_in (int): Input dimension
            head_structure (dict[str, int]): Dictionary mapping head names to their number of output classes
                e.g. {"contrast_agent": 1, "main_structure": 5, "stent_presence": 1}
        """
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.heads = nn.ModuleDict()
        for head_name, num_classes in head_structure.items():
            if not isinstance(num_classes, int) or num_classes < 1:
                raise ValueError(f"Invalid number of classes for head {head_name}: {num_classes}")
            self.heads[head_name] = nn.Sequential(RegressionHead(dim_in, num_classes))

        self.head_structure = head_structure

    def forward(self, x):
        """
        Forward pass through all heads.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim_in]

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping head names to their outputs
        """
        x = self.dropout(x)
        return {head_name: head_module(x) for head_name, head_module in self.heads.items()}


def replace_final_layer(model, config):
    """
    Replace the final layer of the model based on the problem type.
    """
    if hasattr(model, "head"):
        if isinstance(model.head, nn.Sequential):
            in_features = model.head[-1].in_features
        elif isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
        else:
            in_features = model.head.fc1.in_channels if hasattr(model.head, "fc1") else None

        if in_features is None:
            raise AttributeError("Unable to determine input features for the final layer")

        if "mvit" in config["model_name"].lower():
            num_classes = 1 if config["task"] == "regression" else config["num_classes"]
            model.head = TransformerHead(dim_in=in_features, num_classes=num_classes)
            print(
                f"Replaced final layer with TransformerHead(dim_in={in_features}, num_classes={num_classes}) for {config['task']}"
            )
        elif "swin3d" in config["model_name"].lower():
            model.head = nn.Linear(
                in_features=in_features, out_features=config["num_classes"], bias=True
            )
            print(
                f"Replaced final layer with Linear(in_features={in_features}, out_features={config['num_classes']}, bias=True) for {config['task']}"
            )
        else:
            if config["task"] == "regression":
                model.head = RegressionHead(dim_in=in_features, num_classes=1)
                print(
                    f"Replaced final layer with RegressionHead(dim_in={in_features}, num_classes=1) for regression"
                )
            elif config["task"] == "classification":
                if "head_structure" not in config:
                    raise ValueError(
                        "head_structure must be defined in config for multi_head task"
                    )
                model.head = MultiOutputHead(
                    dim_in=in_features, head_structure=config["head_structure"]
                )
                print(
                    f"Replaced final layer with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']})"
                )
            else:
                raise ValueError(f"Unsupported task: {config['task']}")
    elif hasattr(model, "blocks") and hasattr(model.blocks[-1], "proj"):
        # Check and print in_features or shape of the first block
        if hasattr(model.blocks[0], "proj"):
            print(f"in_features of model.blocks[0]: {model.blocks[0].proj.in_features}")
        else:
            print(f"model.blocks[0] does not have 'proj' attribute")

        # Print the in_features of the last block's projection layer
        in_features_last_block = model.blocks[-1].proj.in_features
        print(f"in_features of model.blocks[-1]: {in_features_last_block}")

        # Determine the input features for the new head based on the model type
        if "x3d" in config.get("model_name", ""):
            in_features = 192  # This value is specific for x3d models
        else:
            in_features = in_features_last_block

        # Replace the last block with the appropriate head based on the task
        if config["task"] == "regression":
            model.blocks[-1] = RegressionHead(dim_in=in_features, num_classes=1)
            print(
                f"Replaced final block with RegressionHead(dim_in={in_features}, num_classes=1) for regression"
            )
        elif config["task"] == "classification":
            if "head_structure" not in config:
                raise ValueError("head_structure must be defined in config for multi_head task")
            model.blocks[-1] = MultiOutputHead(
                dim_in=in_features, head_structure=config["head_structure"]
            )
            print(
                f"Replaced final block with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']})"
            )
        else:
            raise ValueError(f"Unsupported task: {config['task']}")
    elif hasattr(model, "norm") and isinstance(model.norm, nn.LayerNorm):
        in_features = model.head[1].in_features

        if config["task"] == "regression":
            model.head = RegressionHead(dim_in=in_features, num_classes=1)
            print(
                f"Replaced final layer with RegressionHead(dim_in={in_features}, num_classes=1) for regression"
            )
        elif config["task"] == "classification":
            if "head_structure" not in config:
                raise ValueError("head_structure must be defined in config for multi_head task")
            model.head = MultiOutputHead(
                dim_in=in_features, head_structure=config["head_structure"]
            )
            print(
                f"Replaced final layer with MultiOutputHead(dim_in={in_features}, head_structure={config['head_structure']})"
            )
        else:
            raise ValueError(f"Unsupported task: {config['task']}")
    else:
        raise AttributeError("Unable to determine input features for the final layer")

    return model


def load_and_modify_model(config):
    """
    Load a model and modify its final layer based on the configuration.
    """
    if "num_classes" not in config and "head_structure" not in config:
        raise ValueError("num_classes or head_structure must be defined in config")
    num_classes = config["num_classes"] if "num_classes" in config else 1
    resize = config.get("resize", 256)
    num_frames = config.get("frames", 64)
    print(f"model_name: {config['model_name']}")
    if config["model_name"] == "x3d_multi":
        model = x3d_multi(num_classes, **config)
    elif config["model_name"] == "timesformer":
        model = timesformer(num_classes, resize, **config)
    elif config["model_name"] == "vivit":
        model = vivit(num_classes, resize, num_frames, **config)
    elif config["model_name"] == "stam":
        model = stam(num_classes, resize, **config)
    elif config["model_name"] in [
        "c2d_r50",
        "i3d_r50",
        "slow_r50",
        "slowfast_r50",
        "slowfast_r101",
        "slowfast_16x8_r101_50_50",
        "csn_r101",
        "r2plus1d_r50",
        "x3d_xs",
        "x3d_s",
        "x3d_m",
        "x3d_l",
        "efficient_x3d_xs",
        "efficient_x3d_s",
        "swin3d_s",
        "swin3d_b",
    ]:
        import torchvision

        if config["model_name"] in ["swin3d_s", "swin3d_b"]:
            model = getattr(torchvision.models.video, config["model_name"])(
                weights="KINETICS400_V1"
            )
        else:
            print(config["model_name"])
            logging.basicConfig(level=logging.DEBUG)
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", config["model_name"], pretrained=True
            )

        # Replace the final layer only for these models
        model = replace_final_layer(model, config)
    elif config["model_name"] == "mvit_v1_b":
        from torchvision.models.video import mvit_v1_b

        model = mvit_v1_b(weights="KINETICS400_V1")
        model = replace_final_layer(model, config)
    elif config["model_name"] == "mvit_v2_s":
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        model = replace_final_layer(model, config)
    elif config["model_name"] == "videopairclassifier":
        if "feature_extractor_backbone" not in config:
            raise ValueError("feature_extractor_backbone must be defined")

        # Get head structure from config
        head_structure = config.get("head_structure", None)

        model = VideoPairClassifier(
            head_structure=head_structure,  # Pass head structure
            feature_extractor_backbone=config["feature_extractor_backbone"],
            checkpoint_path=config.get("feature_extractor_checkpoint_path", None),
            num_frames=config.get("frames", 64),
        )
    else:
        raise ValueError(f"Unsupported model: {config['model_name']}")
    return model
