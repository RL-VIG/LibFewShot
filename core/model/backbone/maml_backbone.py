import torch
import torch.nn as nn
import torch.nn.functional as F

"""
https://github.com/wyharveychen/CloserLookFewShot
"""
class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class Conv64F_fw(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
        Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

        Input:  3 * 84 *84
        Output: 64 * 5 * 5
    """

    def __init__(self, is_flatten=False, is_feature=False):
        super(Conv64F_fw, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.layer1 = nn.Sequential(
            Conv2d_fw(3, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d_fw(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer2 = nn.Sequential(
            Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d_fw(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer3 = nn.Sequential(
            Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d_fw(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer4 = nn.Sequential(
            Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d_fw(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)

        if self.is_feature:
            return out1, out2, out3, out4

        return out4


class Conv32F_fw(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
        Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

        Input:  3 * 84 *84
        Output: 32 * 5 * 5
    """

    def __init__(self, is_flatten=False, is_feature=False):
        super(Conv32F_fw, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.layer1 = nn.Sequential(
            Conv2d_fw(3, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d_fw(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer2 = nn.Sequential(
            Conv2d_fw(32, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d_fw(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer3 = nn.Sequential(
            Conv2d_fw(32, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d_fw(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer4 = nn.Sequential(
            Conv2d_fw(32, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d_fw(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)

        if self.is_feature:
            return out1, out2, out3, out4

        return out4
