import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """FROM SLOWFAST"""

    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SwishEfficient.apply(x)


class SwishEfficient(torch.autograd.Function):
    """FROM SLOWFAST"""

    """Swish activation function: x * sigmoid(x)."""

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
        self.swish = Swish()  # nn.Hardswish()
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
        #         pdb.set_trace()
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # Squeeze-and-Excitation
        if self.index % 2 == 0:
            se_w = self.global_pool(out)
            se_w = self.fc1(se_w)
            se_w = self.relu(se_w)
            se_w = self.fc2(se_w)
            se_w = self.sigmoid(se_w)
            out = out * se_w
        out = self.swish(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class X3D_multi(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        shortcut_type="B",
        widen_factor=1.0,
        dropout=0.5,
        num_classes=1,
        base_bn_splits=8,
        task="class",
    ):
        super().__init__()

        block_inplanes = [
            (int(x * widen_factor), int(y * widen_factor)) for x, y in block_inplanes
        ]
        self.num_classes = num_classes
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
        self.conv6 = nn.Conv2d(
            432,
            64,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False
            #             groups=3
        )  # convolve over third dimension. nchw -> convolve h, T = output

        self.conv7 = nn.Conv3d(
            4096,
            8192,
            kernel_size=(3, 3, 3),  # might want to make (1, 1)
            padding=(1, 1, 1),
            bias=False
            #             groups=3
        )  # convolve over third dimension. nchw -> convolve h, T = output
        self.bn5 = nn.BatchNorm3d(block_inplanes[3][0])
        if task == "class":
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif task == "loc":
            self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc1 = nn.Conv3d(768, 2048, bias=False, kernel_size=1, stride=1)
        if self.num_classes == 2:
            self.fc2 = nn.Linear(1536, 1)
        else:
            self.fc2 = nn.Linear(1536, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes[1]:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block, planes=planes[1], stride=stride
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes[1], stride), nn.BatchNorm3d(planes[1])
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
                    self.in_planes, planes, index=self.index, base_bn_splits=self.base_bn_splits
                )
            )
            self.index += 1

        self.index = 0
        return nn.Sequential(*layers)

    def replace_logits(self, num_classes):
        self.fc2 = nn.Linear(2048, num_classes)

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

    def forward(self, x):
        x = torch.split(x, 1, dim=1)

        for i in range(len(x)):
            z = torch.squeeze(x[i], 1)
            print(z.shape)
            z = z.float()
            z = self.conv1_s(z)
            z = self.conv1_t(z)
            z = self.bn1(z)
            z = self.relu(z)

            z = self.layer1(z)
            z = self.layer2(z)
            z = self.layer3(z)
            z = self.layer4(z)

            z = self.conv5(z)
            z = self.bn5(z)
            z = self.relu(z)
            if i == 1:
                z1 = z
                print("z1 shape", z1.shape)
                z_combined = torch.stack(
                    [z0, z1], dim=2
                )  # change variable name to avoid confusion
                fx_ntvu = torch.flatten(z_combined, start_dim=3, end_dim=5)
                fy_ntvu = self.conv6(fx_ntvu)  # convolve over v
                fy_ntvchw = fy_ntvu.reshape(
                    torch.Size(
                        [
                            z_combined.shape[0],
                            64,
                            z_combined.shape[2],
                            z_combined.shape[3],
                            z_combined.shape[4],
                            z_combined.shape[5],
                        ]
                    )
                )

                fy_ntcvhw = fy_ntvchw.permute((0, 1, 3, 2, 4, 5))
                fy_nuvhw = torch.flatten(fy_ntcvhw, start_dim=1, end_dim=2)
                yfy_nuvhw = self.conv7(fy_nuvhw)
                print(fy_ntcvhw.shape)
                yfy_ntcvhw = yfy_nuvhw.reshape(
                    torch.Size(
                        [
                            fy_ntcvhw.shape[0],
                            512,
                            fy_ntcvhw.shape[2],
                            fy_ntcvhw.shape[3],
                            fy_ntcvhw.shape[4],
                            fy_ntcvhw.shape[5],
                        ]
                    )
                )
                yfy_ntvchw = yfy_ntcvhw.permute((0, 1, 3, 2, 4, 5))

                yfy_nuchw = torch.flatten(yfy_ntvchw, start_dim=1, end_dim=2)
                z = self.avgpool(yfy_nuchw)
                z = self.relu(z)
                if self.task == "class":
                    z = z.squeeze(4).squeeze(3).squeeze(2)  # B C
                    z = self.dropout(z)
                    z = self.fc2(z).unsqueeze(2)  # B C 1
                if self.num_classes == 2:
                    z = self.sigmoid(z)
            elif i == 2:
                z2 = z
                z2 = torch.stack([z0, z1, z2], dim=2)
                fx_ntvu = torch.flatten(z2, start_dim=3, end_dim=5)
                fy_ntvu = self.conv6(fx_ntvu)  # convolve over v
                fy_ntvchw = fy_ntvu.reshape(
                    torch.Size(
                        [z2.shape[0], 64, z2.shape[2], z2.shape[3], z2.shape[4], z2.shape[5]]
                    )
                )

                fy_ntcvhw = fy_ntvchw.permute((0, 1, 3, 2, 4, 5))
                fy_nuvhw = torch.flatten(fy_ntcvhw, start_dim=1, end_dim=2)
                yfy_nuvhw = self.conv7(fy_nuvhw)
                yfy_ntcvhw = yfy_nuvhw.reshape(
                    torch.Size(
                        [
                            fy_ntcvhw.shape[0],
                            512,
                            fy_ntcvhw.shape[2],
                            fy_ntcvhw.shape[3],
                            fy_ntcvhw.shape[4],
                            fy_ntcvhw.shape[5],
                        ]
                    )
                )
                yfy_ntvchw = yfy_ntcvhw.permute((0, 1, 3, 2, 4, 5))

                yfy_nuchw = torch.flatten(yfy_ntvchw, start_dim=1, end_dim=2)
                z = self.avgpool(yfy_nuchw)
                #         z = self.fc1(z)
                z = self.relu(z)
                if self.task == "class":
                    z = z.squeeze(4).squeeze(3).squeeze(2)  # B C
                    z = self.dropout(z)
                    z = self.fc2(z).unsqueeze(2)  # B C 1
                if self.num_classes == 2:
                    z = self.sigmoid(z)
            else:
                z0 = z

        return z
