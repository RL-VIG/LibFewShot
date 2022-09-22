# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.modules.utils import _pair


class _ConvNdMtl(nn.Module):
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
            self.weight = nn.Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size)
            )
            self.mtl_weight = nn.Parameter(
                torch.ones(in_channels, out_channels // groups, 1, 1)
            )
        else:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            )
            self.mtl_weight = nn.Parameter(
                torch.ones(out_channels, in_channels // groups, 1, 1)
            )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.mtl_bias = nn.Parameter(torch.zeros(out_channels))
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


def convert_mtl_module(module, MTL=False):
    """Convert a normal model to MTL model.

    Replace nn.Conv2d with Conv2dMtl.

    Args:
        module: The module (model component) to be converted.

    Returns: A MTL model.

    """
    module_output = module
    if isinstance(module, torch.nn.modules.Conv2d):
        module_output = Conv2dMtl(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            False if module.bias is None else True,
            MTL,
        )

    for name, child in module.named_children():
        module_output.add_module(name, convert_mtl_module(child, MTL))
    del module
    return module_output
