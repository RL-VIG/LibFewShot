# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/wyharveychen/CloserLookFewShot.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# FIXME bias=False
class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(
                x, self.weight.fast, self.bias.fast
            )  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


# FIXME add complete parameters support
class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
    ):
        super(Conv2d_fw, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    None,
                    stride=self.stride,
                    padding=self.padding,
                )
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    self.bias.fast,
                    stride=self.stride,
                    padding=self.padding,
                )
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


# FIXME add complete parameter support
class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight.fast,
                self.bias.fast,
                training=True,
                momentum=1,
            )
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                training=True,
                momentum=1,
            )
        return out


def convert_maml_module(module):
    """Convert a normal model to MAML model.

    Replace nn.Linear with Linear_fw, nn.Conv2d with Conv2d_fw.

    Args:
        module: The module (model component) to be converted.

    Returns: A MAML model.

    """
    module_output = module
    if isinstance(module, torch.nn.modules.Linear):
        module_output = Linear_fw(
            module.in_features,
            module.out_features,
            False if module.bias is None else True,
        )
    elif isinstance(module, torch.nn.modules.Conv2d):
        module_output = Conv2d_fw(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            False if module.bias is None else True,
        )
    elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        module_output = BatchNorm2d_fw(
            module.num_features,
        )

    for name, child in module.named_children():
        module_output.add_module(name, convert_maml_module(child))
    del module
    return module_output
