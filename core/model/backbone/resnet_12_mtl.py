import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.backbone.dropblock import DropBlock

# this file is a bit diferent from resnet_12_mtl:
# resnet_12_mtl is the origin backbone which is used in MTL's official implementation.
# resnet_12_ori_mtl is adapted from resnet12.py, use the same channels([64,160,320,640]) or other settings in the
# origin resnet12. this backbone is used in table.2, tabel.3 and tabel.4 for equal comparisons.

# resnet_12 backbone for 'Meta-Transfer Learning for Few-Shot Learning' with learnable scale and shift
# 基于resnet12，我们自己修改的代码

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair


class _ConvNdMtl(Module):
    """The class for meta-transfer convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, MTL):
        super(_ConvNdMtl, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
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
            self.weight = Parameter(torch.Tensor(
                    in_channels, out_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(in_channels, out_channels // groups, 1, 1))
        else:
            self.weight = Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(out_channels, in_channels // groups, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.mtl_bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mtl_bias', None)
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
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.mtl_bias.data.uniform_(0, 0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.MTL is not None:
            s += ', MTL={MTL}'
        return s.format(**self.__dict__)


class Conv2dMtl(_ConvNdMtl):
    """The class for meta-transfer convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, MTL=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.MTL = MTL
        super(Conv2dMtl, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias, MTL)

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
        return F.conv2d(inp, new_weight, new_bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv3x3MTL(in_planes, out_planes, stride=1, MTL=False):
    """3x3 convolution with padding"""
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, MTL=MTL)


class BasicBlockMTL(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0,
                 drop_block=False, block_size=1, MTL=False):
        super(BasicBlockMTL, self).__init__()
        self.conv1 = conv3x3MTL(inplanes, planes,MTL=MTL)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3MTL(planes, planes, MTL=MTL)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3MTL(planes, planes, MTL=MTL)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(
                        1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked),
                        1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (
                        feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training,
                                inplace=True)

        return out


class ResNet12MTLORI(nn.Module):

    def __init__(self, block=BasicBlockMTL, keep_prob=1.0, avg_pool=True, drop_rate=0.1,
                 dropblock_size=5, is_flatten=True, MTL=False):
        self.inplanes = 3
        super(ResNet12MTLORI, self).__init__()
        self.Conv2d = Conv2dMtl
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate, MTL=MTL)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate, MTL=MTL)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate,
                                       drop_block=True, block_size=dropblock_size, MTL=MTL)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate,
                                       drop_block=True, block_size=dropblock_size, MTL=MTL)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.is_flatten = is_flatten
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False,
                    block_size=1, MTL=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    self.Conv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=1, bias=False,MTL=MTL),
                    nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
                block(self.inplanes, planes, stride, downsample,MTL=MTL))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        if self.is_flatten:
            x = x.view(x.size(0), -1)
        return x


def resnet12mtlori(keep_prob=1.0, avg_pool=True, is_flatten=True,MTL=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet12MTLORI(BasicBlockMTL, keep_prob=keep_prob, avg_pool=avg_pool, is_flatten=is_flatten, MTL=MTL, **kwargs)
    return model
