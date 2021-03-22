from torch import nn
import torch


class Conv64F(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
        Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

        Input:  3 * 84 *84
        Output: 64 * 5 * 5
    """

    def __init__(self, is_flatten=False, is_feature=False):
        super(Conv64F, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
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

        # out4 = torch.nn.functional.adaptive_avg_pool2d(out4, (1,1)).squeeze()

        return out4


class Conv64FLeakyReLU(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a LeakyReLU layer and a Maxpooling layer.
        Used in the original DN4 and CovaMNet: https://github.com/WenbinLee/DN4.git.

        Input:  3 * 84 *84
        Output: 64 * 21 * 21
    """

    def __init__(self, is_flatten=False, is_feature=False):
        super(Conv64FLeakyReLU, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True), )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True), )

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


class Conv64FReLU(nn.Module):
    """
    https://github.com/floodsung/LearningToCompare_FSL/blob/master/miniimagenet/miniimagenet_train_few_shot.py
    """

    def __init__(self, is_flatten=False, is_feature=False):
        super(Conv64FReLU, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True))

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


def R2D2_conv_block(in_channels, out_channels, retain_activation=True, keep_prob=1.0, pool_stride=2):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2, stride=pool_stride)
    )
    if retain_activation:
        block.add_module("LeakyReLU", nn.LeakyReLU(0.1))

    if keep_prob < 1.0:
        block.add_module("Dropout", nn.Dropout(p=1 - keep_prob, inplace=False))

    return block


class R2D2Embedding(nn.Module):
    def __init__(self, x_dim=3, h1_dim=96, h2_dim=192, h3_dim=384, z_dim=512,
                 retain_last_activation=True):
        super(R2D2Embedding, self).__init__()

        self.block1 = R2D2_conv_block(x_dim, h1_dim)
        self.block2 = R2D2_conv_block(h1_dim, h2_dim)
        self.block3 = R2D2_conv_block(h2_dim, h3_dim, keep_prob=0.9)
        # In the last conv block, we disable activation function to boost the classification accuracy.
        # This trick was proposed by Gidaris et al. (CVPR 2018).
        # With this trick, the accuracy goes up from 50% to 51%.
        # Although the authors of R2D2 did not mention this trick in the paper,
        # we were unable to reproduce the result of Bertinetto et al. without resorting to this trick.
        self.block4 = R2D2_conv_block(h3_dim, z_dim, retain_activation=retain_last_activation, keep_prob=0.9,
                                      pool_stride=1)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        # Flatten and concatenate the output of the 3rd and 4th conv blocks as proposed in R2D2 paper.
        return torch.cat((b3.view(b3.size(0), -1), b4.view(b4.size(0), -1)), 1)