from torch import nn
import torch


class Conv32F(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
        Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

        Input:  3 * 84 *84
        Output: 32 * 5 * 5
    """

    def __init__(self, is_flatten=False, is_feature=False):
        super(Conv32F, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
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


class Conv32FLeakyReLU(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a LeakyReLU layer and a Maxpooling layer.
        Used in the original DN4 and CovaMNet: https://github.com/WenbinLee/DN4.git.

        Input:  3 * 84 *84
        Output: 32 * 21 * 21
    """

    def __init__(self, is_flatten=False, is_feature=False):
        super(Conv32FLeakyReLU, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
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


class Conv32FReLU(nn.Module):
    """
    https://github.com/floodsung/LearningToCompare_FSL/blob/master/miniimagenet/miniimagenet_train_few_shot.py
    """

    def __init__(self, is_flatten=False, is_feature=False):
        super(Conv32FReLU, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(inplace=True),
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
