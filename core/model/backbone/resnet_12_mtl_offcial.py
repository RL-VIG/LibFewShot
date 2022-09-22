# -*- coding: utf-8 -*-
"""
This ResNet12MTL is a variant of ResNet12.
Official implementation of 'Meta-Transfer Learning for Few-Shot Learning' with learnable scale and shift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair


class _ConvNdMtl(Module):
    """The class for meta-transfer convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        MTL,
    ):
        super(_ConvNdMtl, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.MTL = MTL
        if transposed:
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size)
            )
            self.mtl_weight = Parameter(
                torch.ones(in_channels, out_channels // groups, 1, 1)
            )
        else:
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            )
            self.mtl_weight = Parameter(
                torch.ones(out_channels, in_channels // groups, 1, 1)
            )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.mtl_bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("mtl_bias", None)
        if MTL:
            self.weight.requires_grad = False
            if bias:
                self.bias.requires_grad = False
        else:
            self.mtl_weight.requires_grad = False
            if bias:
                self.mtl_bias.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.mtl_bias.data.uniform_(0, 0)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.MTL is not None:
            s += ", MTL={MTL}"
        return s.format(**self.__dict__)


class Conv2dMtl(_ConvNdMtl):
    """The class for meta-transfer convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        MTL=False,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.MTL = MTL
        super(Conv2dMtl, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            MTL,
        )

    def forward(self, inp):  # override conv2d forward
        if self.MTL:
            new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
            new_weight = self.weight.mul(new_mtl_weight)
            if self.bias is not None:
                new_bias = self.bias + self.mtl_bias
            else:
                new_bias = None
        else:
            new_weight = self.weight
            new_bias = self.bias
        return F.conv2d(
            inp,
            new_weight,
            new_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def conv3x3MTL(in_planes, out_planes, stride=1, MTL=False):
    """3x3 convolution with padding"""
    return Conv2dMtl(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        MTL=MTL,
    )


class BasicBlockMTL(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, MTL=False):
        super(BasicBlockMTL, self).__init__()
        self.conv1 = conv3x3MTL(inplanes, planes, stride, MTL=MTL)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3MTL(planes, planes, MTL=MTL)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class ResNetMTLOfficial(nn.Module):
    def __init__(self, MTL=False):
        super(ResNetMTLOfficial, self).__init__()
        self.Conv2d = Conv2dMtl
        block = BasicBlockMTL
        self.inplanes = iChannels = 80
        self.conv1 = self.Conv2d(
            3, iChannels, kernel_size=3, stride=1, padding=1, MTL=MTL
        )
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 160, 4, stride=2, MTL=MTL)
        self.layer2 = self._make_layer(block, 320, 4, stride=2, MTL=MTL)
        self.layer3 = self._make_layer(block, 640, 4, stride=2, MTL=MTL)
        self.avgpool = nn.AvgPool2d(10, stride=1)

        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, MTL=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    MTL=MTL,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, MTL=MTL))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, MTL=MTL))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x


def resnet12MTLofficial(**kwargs):
    """Constructs a ResNet-12 model."""
    model = ResNetMTLOfficial(**kwargs)
    return model


if __name__ == "__main__":
    model = resnet12MTLofficial().cuda()
    data = torch.rand(10, 3, 84, 84).cuda()
    output = model(data)
    print(output.size())
