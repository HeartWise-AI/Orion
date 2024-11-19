import torch
import torch.nn as nn
import torchvision.models.video
from pytorchvideo.models.accelerator.mobile_cpu.efficient_x3d import (
    create_x3d as create_efficient_x3d,
)
from pytorchvideo.models.csn import create_csn
from pytorchvideo.models.r2plus1d import create_r2plus1d
from pytorchvideo.models.resnet import create_resnet
from pytorchvideo.models.slowfast import create_slowfast

# from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers
from pytorchvideo.models.x3d import create_x3d


def get_model(model_name, kwargs):
    """ "
    Description: Get a model for video classification.
    Input:
    * model_name: (string) Model name to get. Can be ["r2plus1d", "slowfast", "resnet", "csn", "x3d", "efficient_x3d", "mvit"]
    * kwargs: (dict) Parameters of the model. Refer to https://github.com/facebookresearch/pytorchvideo/tree/main/pytorchvideo/models to know the possible parameters of every model.
    Output: Model.
    """

    if model_name == "r2plus1d":
        model = create_r2plus1d(**kwargs)
    elif model_name == "slowfast":
        model = create_slowfast(**kwargs)
    elif model_name == "resnet":
        model = create_resnet(**kwargs)
    elif model_name == "csn":
        model = create_csn(**kwargs)
    elif model_name == "x3d":
        model = create_x3d(**kwargs)

    elif model_name == "efficient_x3d":
        model = create_efficient_x3d(**kwargs)
    elif model_name == "mvit":
        model = create_multiscale_vision_transformers(**kwargs)
    else:
        assert 1 == 0, "Model not found."
    return model


# However, in many deep learning frameworks like PyTorch, the activation function is not explicitly included in the model's architecture for classification tasks. Instead, it's integrated into the loss function during training. For example:
def get_last_layer_name(model):
    last_layer_name = None
    for name, module in model.named_modules():
        last_layer_name = name
    return last_layer_name


# or multi-class classification, you often use the CrossEntropyLoss in PyTorch, which internally applies the softmax function.
# For binary classification, you might use BCEWithLogitsLoss, which applies the sigmoid function.
# Therefore, while the model's architecture ends with a linear layer, the softmax or sigmoid activation is still applied during training and inference, but it's just handled differently than being a direct part of the sequential model layers. This approach is computationally efficient and also helps with numerical stability.


def get_pretrained_model(model_name, num_classes=400, task="classification"):
    """
    Description: Get pretrained model on Kinetics-400 for video classification.
    Input:
    * model_name: (string) Model name to be pretrained. Can be ["c2d_r50", "i3d_r50", "slow_r50","slowfast_r50", "slowfast_r101", "slowfast_16x8_r101_50_50", "csn_r101", "r2plus1d_r50", "x3d_xs", "x3d_s", "x3d_m", "x3d_l", "mvit_base_16x4", "mvit_base_32x3", "efficient_x3d_xs", "efficient_x3d_s"]
    * num_classes: (int) Number of classes for classification.
    Output: Pretrained model.
    """

    param = {}
    param["c2d_r50"] = {}
    param["c2d_r50"]["model_num_classes"] = num_classes
    param["c2d_r50"]["stem_conv_kernel_size"] = (1, 7, 7)
    param["c2d_r50"]["stage1_pool"] = nn.MaxPool3d
    param["c2d_r50"]["stage_conv_a_kernel_size"] = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    )
    # param["c2d_r50"]["activation"] = nn.Softmax

    param["i3d_r50"] = {}
    param["i3d_r50"]["model_num_classes"] = num_classes
    param["i3d_r50"]["stem_conv_kernel_size"] = (5, 7, 7)
    param["i3d_r50"]["stage1_pool"] = nn.MaxPool3d
    param["i3d_r50"]["stage_conv_a_kernel_size"] = (
        (3, 1, 1),
        [(3, 1, 1), (1, 1, 1)],
        [(3, 1, 1), (1, 1, 1)],
        [(1, 1, 1), (3, 1, 1)],
    )
    # param["i3d_r50"]["activation"] = nn.Softmax

    param["slow_r50"] = {}
    param["slow_r50"]["model_num_classes"] = num_classes
    param["slow_r50"]["stem_conv_kernel_size"] = (1, 7, 7)
    param["slow_r50"]["head_pool_kernel_size"] = (8, 7, 7)
    param["slow_r50"]["model_depth"] = 50
    # param["slow_r50"]["activation"] = nn.Softmax

    param["slowfast_r50"] = {}
    param["slowfast_r50"]["model_num_classes"] = num_classes
    param["slowfast_r50"]["model_depth"] = 50
    param["slowfast_r50"]["slowfast_fusion_conv_kernel_size"] = (7, 1, 1)
    # param["slowfast_r50"]["activation"] = nn.Softmax

    param["slowfast_r101"] = {}
    param["slowfast_r101"]["model_num_classes"] = num_classes
    param["slowfast_r101"]["model_depth"] = 101
    param["slowfast_r101"]["slowfast_fusion_conv_kernel_size"] = (5, 1, 1)
    # param["slowfast_r101"]["activation"] = nn.Softmax

    param["slowfast_16x8_r101_50_50"] = {}
    param["slowfast_16x8_r101_50_50"]["model_num_classes"] = num_classes
    param["slowfast_16x8_r101_50_50"]["model_depth"] = 101
    param["slowfast_16x8_r101_50_50"]["slowfast_fusion_conv_kernel_size"] = (5, 1, 1)
    param["slowfast_16x8_r101_50_50"]["stage_conv_a_kernel_sizes"] = (
        (
            (1, 1, 1),
            (1, 1, 1),
            ((3, 1, 1),) * 6 + ((1, 1, 1),) * (23 - 6),
            (3, 1, 1),
        ),
        (
            (3, 1, 1),
            (3, 1, 1),
            ((3, 1, 1),) * 6 + ((1, 1, 1),) * (23 - 6),
            (3, 1, 1),
        ),
    )
    param["slowfast_16x8_r101_50_50"]["head_pool_kernel_sizes"] = ((16, 7, 7), (64, 7, 7))
    # param["slowfast_16x8_r101_50_50"]["activation"] = nn.Softmax

    param["csn_r101"] = {}
    param["csn_r101"]["model_num_classes"] = num_classes
    param["csn_r101"]["model_depth"] = 101
    param["csn_r101"]["stem_pool"] = nn.MaxPool3d
    param["csn_r101"]["head_pool_kernel_size"] = (4, 7, 7)
    # param["csn_r101"]["activation"] = nn.Softmax

    param["r2plus1d_r50"] = {}
    param["r2plus1d_r50"]["model_num_classes"] = num_classes
    param["r2plus1d_r50"]["dropout_rate"] = 0.5

    param["x3d_xs"] = {}
    param["x3d_xs"]["model_num_classes"] = num_classes
    param["x3d_xs"]["input_clip_length"] = 4
    param["x3d_xs"]["input_crop_size"] = 160

    param["x3d_s"] = {}
    param["x3d_s"]["model_num_classes"] = num_classes
    param["x3d_s"]["input_clip_length"] = 13
    param["x3d_s"]["input_crop_size"] = 160

    param["x3d_m"] = {}
    param["x3d_m"]["model_num_classes"] = num_classes
    param["x3d_m"]["input_clip_length"] = 16
    param["x3d_m"]["input_crop_size"] = 224

    param["x3d_l"] = {}
    param["x3d_l"]["model_num_classes"] = num_classes
    param["x3d_l"]["input_clip_length"] = 16
    param["x3d_l"]["input_crop_size"] = 312
    param["x3d_l"]["depth_factor"] = 5.0

    param["mvit_base_16x4"] = {}
    param["mvit_base_16x4"]["head_num_classeses"] = num_classes
    param["mvit_base_16x4"]["spatial_size"] = 224
    param["mvit_base_16x4"]["temporal_size"] = 16
    param["mvit_base_16x4"]["embed_dim_mul"] = [[1, 2.0], [3, 2.0], [14, 2.0]]
    param["mvit_base_16x4"]["atten_head_mul"] = [[1, 2.0], [3, 2.0], [14, 2.0]]
    param["mvit_base_16x4"]["pool_q_stride_size"] = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
    param["mvit_base_16x4"]["pool_kv_stride_adaptive"] = [1, 8, 8]
    param["mvit_base_16x4"]["pool_kvq_kernel"] = [3, 3, 3]

    param["mvit_base_32x3"] = {}
    param["mvit_base_32x3"]["head_num_classeses"] = num_classes
    param["mvit_base_32x3"]["spatial_size"] = 224
    param["mvit_base_32x3"]["temporal_size"] = 32
    param["mvit_base_32x3"]["embed_dim_mul"] = [[1, 2.0], [3, 2.0], [14, 2.0]]
    param["mvit_base_32x3"]["atten_head_mul"] = [[1, 2.0], [3, 2.0], [14, 2.0]]
    param["mvit_base_32x3"]["pool_q_stride_size"] = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
    param["mvit_base_32x3"]["pool_kv_stride_adaptive"] = [1, 8, 8]
    param["mvit_base_32x3"]["pool_kvq_kernel"] = [3, 3, 3]

    param["efficient_x3d_xs"] = {}
    param["efficient_x3d_xs"]["num_classeses"] = num_classes
    param["efficient_x3d_xs"]["expansion"] = "XS"

    param["efficient_x3d_s"] = {}
    param["efficient_x3d_s"]["num_classeses"] = num_classes
    param["efficient_x3d_s"]["expansion"] = "S"

    param["swin3d_s"] = {}

    param["swin3d_b"] = {}

    kwargs = param[model_name]

    if model_name == "c2d_r50":
        model = get_model("resnet", kwargs)
    elif model_name == "i3d_r50":
        model = get_model("resnet", kwargs)
    elif model_name == "slow_r50":
        model = get_model("resnet", kwargs)
    elif model_name == "slowfast_r50":
        model = get_model("slowfast", kwargs)
    elif model_name == "slowfast_r101":
        model = get_model("slowfast", kwargs)
    elif model_name == "slowfast_16x8_r101_50_50":
        model = get_model("slowfast", kwargs)
    elif model_name == "csn_r101":
        model = get_model("csn", kwargs)
    elif model_name == "r2plus1d_r50":
        model = get_model("r2plus1d", kwargs)
    elif model_name == "x3d":
        model = get_model("x3d", kwargs)
        last_layer_name = get_last_layer_name(model)
        print("The last layer is:", last_layer_name)

        # Modify this part to match the swin3d approach
        if task == "classification":
            new_linear_layer = nn.Linear(in_features=400, out_features=num_classes)
            model.blocks[5].proj = new_linear_layer
            model.blocks[5].activation = nn.Identity()  # Remove any existing activation
        elif task == "regression":
            new_linear_layer = nn.Linear(in_features=400, out_features=1)
            model.blocks[5].proj = new_linear_layer
            model.blocks[5].activation = nn.Identity()  # Remove any existing activation

    elif model_name == "x3d_xs":
        model = get_model("x3d", kwargs)
    elif model_name == "x3d_s":
        model = get_model("x3d", kwargs)
    elif model_name == "x3d_m":
        model = get_model("x3d", kwargs)
    elif model_name == "x3d_l":
        model = get_model("x3d", kwargs)
    elif model_name == "mvit_base_16x4":
        model = get_model("mvit", kwargs)
    elif model_name == "mvit_base_32x3":
        model = get_model("mvit", kwargs)
    elif model_name == "efficient_x3d_xs":
        model = get_model("efficient_x3d", kwargs)
    elif model_name == "efficient_x3d_s":
        model = get_model("efficient_x3d", kwargs)
    elif model_name == "swin3d_s":
        print("Importing swin3d_s for num_classes", num_classes)
        model = torchvision.models.video.swin3d_s(weights="KINETICS400_V1")
        n_inputs = model.head.in_features
        if task == "classification":
            model.head = nn.Linear(n_inputs, num_classes)
        elif task == "regression":
            model.head = nn.Linear(
                n_inputs, 1
            )  # For regression, typically we predict a single value
        # No need to add Softmax or Sigmoid here - Do it during inference

    elif model_name == "swin3d_b":
        print("Importing swin3d_b for num_classes", num_classes)
        model = torchvision.models.video.swin3d_b(weights="KINETICS400_IMAGENET22K_V1")
        n_inputs = model.head.in_features

        if task == "classification":
            model.head = nn.Linear(n_inputs, num_classes)
        elif task == "regression":
            model.head = nn.Linear(
                n_inputs, 1
            )  # For regression, typically we predict a single value

        # No need to add Softmax or Sigmoid here - Do it during inference

    else:
        assert False, "Pretrained model not found."

    if model_name not in ["swin3d_s", "swin3d_b"]:
        model_dict = model.state_dict()
        pretrained_dict = torch.hub.load(
            "facebookresearch/pytorchvideo", model_name, pretrained=True
        ).state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in list(model_dict.keys()) and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if task == "classification":
            if model_name in [
                "c2d_r50",
                "i3d_r50",
                "slow_r50",
                "csn_r101",
                "r2plus1d_r50",
                "slowfast_r50",
                "slowfast_r101",
                "slowfast_16x8_r101_50_50",
            ]:
                model.blocks[-1].activation = nn.Softmax(1)
            elif model_name in ["mvit_base_16x4", "mvit_base_32x3"]:
                model.head = nn.Sequential(model.head, nn.Softmax(1))
        if task == "regression":
            try:
                model.blocks[-1].activation = None
            except:
                print("exception")
                temp = 1
            if model_name in [
                "c2d_r50",
                "i3d_r50",
                "slow_r50",
                "csn_r101",
                "r2plus1d_r50",
                "slowfast_r50",
                "slowfast_r101",
                "slowfast_16x8_r101_50_50",
            ]:
                model.blocks[-1].activation = nn.Sigmoid()
            elif model_name in ["mvit_base_16x4", "mvit_base_32x3"]:
                model.head = nn.Sequential(model.head, nn.Sigmoid())
        # print(model.blocks[-1])

    return model
