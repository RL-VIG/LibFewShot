# -*- coding: utf-8 -*-
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SimpleBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(SimpleBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet10(nn.Module):
    def __init__(
        self,
        out_dims=(64, 128, 256, 512),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        avg_pool=True,
        is_flatten=True,
        is_feature=False,
    ):
        super(ResNet10, self).__init__()
        self.avg_pool = avg_pool
        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        blocks = []
        inplanes = 64
        for planes, stride, dilation in zip(out_dims, strides, dilations):
            blocks.append(SimpleBlock(inplanes, planes, stride, dilation))
            inplanes = planes
        self.layers = nn.ModuleList(blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = []
        out = self.stem(x)
        for layer in self.layers:
            out = layer(out)
            features.append(out)

        if self.avg_pool:
            out = self.avgpool(out)
        if self.is_flatten:
            out = torch.flatten(out, 1)
        if self.is_feature:
            features[-1] = out
            return tuple(features)
        return out

    def convert_state_dict(self, state_dict):
        """Convert the original LDP-Net ResNet10 checkpoint keys."""
        converted = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("trunk.0."):
                new_key = new_key.replace("trunk.0.", "stem.0.", 1)
            elif new_key.startswith("trunk.1."):
                new_key = new_key.replace("trunk.1.", "stem.1.", 1)
            elif new_key.startswith("trunk."):
                parts = new_key.split(".", 2)
                if len(parts) == 3 and parts[1].isdigit():
                    block_idx = int(parts[1]) - 4
                    if 0 <= block_idx < len(self.layers):
                        new_key = "layers.{}.{}".format(block_idx, parts[2])

            new_key = new_key.replace(".C1.", ".conv1.")
            new_key = new_key.replace(".BN1.", ".bn1.")
            new_key = new_key.replace(".C2.", ".conv2.")
            new_key = new_key.replace(".BN2.", ".bn2.")
            new_key = new_key.replace(".shortcut.", ".downsample.0.")
            new_key = new_key.replace(".BNshortcut.", ".downsample.1.")
            converted[new_key] = value
        return converted


def resnet10(**kwargs):
    return ResNet10(**kwargs)
