import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SwishEfficient.apply(x)


class SwishEfficient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(1, stride, stride),
        padding=1,
        bias=False,
        groups=in_planes,
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=(1, stride, stride), bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None, index=0, base_bn_splits=8):
        super().__init__()

        self.index = index
        self.base_bn_splits = base_bn_splits
        self.conv1 = conv1x1x1(in_planes, planes[0])
        self.bn1 = nn.BatchNorm3d(planes[0])
        self.conv2 = conv3x3x3(planes[0], planes[0], stride)
        self.bn2 = nn.BatchNorm3d(planes[0])
        self.conv3 = conv1x1x1(planes[0], planes[1])
        self.bn3 = nn.BatchNorm3d(planes[1])
        self.swish = Swish()
        self.relu = nn.ReLU(inplace=True)
        if self.index % 2 == 0:
            width = self.round_width(planes[0])
            self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc1 = nn.Conv3d(planes[0], width, kernel_size=1, stride=1)
            self.fc2 = nn.Conv3d(width, planes[0], kernel_size=1, stride=1)
            self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def round_width(self, width, multiplier=0.0625, min_width=8, divisor=8):
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.index % 2 == 0:
            se_w = self.global_pool(out)
            se_w = self.fc1(se_w)
            se_w = self.relu(se_w)
            se_w = self.fc2(se_w)
            se_w = self.sigmoid(se_w)
            out = out * se_w

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class X3D_legacy(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        shortcut_type="B",
        widen_factor=1.0,
        dropout=0.5,
        n_classes=24,
        base_bn_splits=8,
        task="classification",
    ):
        super().__init__()

        block_inplanes = [
            (int(x * widen_factor), int(y * widen_factor)) for x, y in block_inplanes
        ]
        self.index = 0
        self.base_bn_splits = base_bn_splits
        self.task = task

        self.in_planes = block_inplanes[0][1]

        self.conv1_s = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
        )
        self.conv1_t = nn.Conv3d(
            self.in_planes,
            self.in_planes,
            kernel_size=(5, 1, 1),
            stride=(1, 1, 1),
            padding=(2, 0, 0),
            bias=False,
            groups=self.in_planes,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type, stride=2
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )
        self.conv5 = nn.Conv3d(
            block_inplanes[3][1],
            block_inplanes[3][0],
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.bn5 = nn.BatchNorm3d(block_inplanes[3][0])

        if task in ["classification", "regression"]:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif task == "loc":
            self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            raise ValueError(f"Unsupported task type: {task}")

        self.fc1 = nn.Conv3d(block_inplanes[3][0], 2048, bias=False, kernel_size=1, stride=1)

        if task == "classification":
            if n_classes <= 2:
                self.fc2 = nn.Linear(2048, 1)
            else:
                self.fc2 = nn.Linear(2048, n_classes)
        elif task == "regression":
            self.regress = nn.Linear(2048, 1)
        elif task == "loc":
            self.fc2 = nn.Linear(2048, n_classes)

        self.dropout = nn.Dropout(dropout)

        self.n_classes = n_classes

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if out.is_cuda:
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.conv1_t(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.avgpool(x)

        x = self.fc1(x)

        if self.task in ["classification", "loc"]:
            x = self.relu(x)
            x = x.squeeze(4).squeeze(3)
            if self.task == "classification":
                x = x.squeeze(2)  # B C
                x = self.dropout(x)
                x = self.fc2(x).unsqueeze(2)  # B C 1
            else:  # loc
                x = x.permute(0, 2, 1)  # B T C
                x = self.dropout(x)
                x = self.fc2(x)
                x = x.permute(0, 2, 1)  # B C T
        elif self.task == "regression":
            x = x.squeeze(4).squeeze(3).permute(0, 2, 1)  # B T C
            x = self.dropout(x)
            x = self.regress(x)
        return x

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes[1]:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block, planes=planes[1], stride=stride
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes[1], stride),
                    nn.BatchNorm3d(planes[1]),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                index=self.index,
                base_bn_splits=self.base_bn_splits,
            )
        )
        self.in_planes = planes[1]
        self.index += 1
        for i in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    index=self.index,
                    base_bn_splits=self.base_bn_splits,
                )
            )
            self.index += 1

        self.index = 0
        return nn.Sequential(*layers)

    def replace_logits(self, n_classes):
        self.fc2 = nn.Linear(2048, n_classes)

    def update_bn_splits_long_cycle(self, long_cycle_bn_scale):
        for m in self.modules():
            if isinstance(m, SubBatchNorm3d):
                m.num_splits = self.base_bn_splits * long_cycle_bn_scale
                m.split_bn = nn.BatchNorm3d(
                    num_features=m.num_features * m.num_splits, affine=False
                ).to(m.weight.device)
        return self.base_bn_splits * long_cycle_bn_scale

    def aggregate_sub_bn_stats(self):
        """find all SubBN modules and aggregate sub-BN stats."""
        count = 0
        for m in self.modules():
            if isinstance(m, SubBatchNorm3d):
                m.aggregate_stats()
                count += 1
        return count
