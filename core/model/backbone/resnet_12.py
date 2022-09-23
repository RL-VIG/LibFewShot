# -*- coding: utf-8 -*-
"""
This ResNet network was designed following the practice of the following papers:
TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.backbone.utils.dropblock import DropBlock


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
        use_pool=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_pool = use_pool

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
        if self.use_pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked),
                    1.0 - self.drop_rate,
                )
                gamma = (
                    (1 - keep_rate)
                    / self.block_size**2
                    * feat_size**2
                    / (feat_size - self.block_size + 1) ** 2
                )
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(
                    out, p=self.drop_rate, training=self.training, inplace=True
                )

        return out


class BasicBlockWithoutResidual(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
        use_pool=True,
    ):
        super(BasicBlockWithoutResidual, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_pool = use_pool

    def forward(self, x):
        self.num_batches_tracked += 1

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)
        if self.use_pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked),
                    1.0 - self.drop_rate,
                )
                gamma = (
                    (1 - keep_rate)
                    / self.block_size**2
                    * feat_size**2
                    / (feat_size - self.block_size + 1) ** 2
                )
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(
                    out, p=self.drop_rate, training=self.training, inplace=True
                )

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        blocks=[BasicBlock, BasicBlock, BasicBlock, BasicBlock],
        planes=[64, 160, 320, 640],
        keep_prob=1.0,
        avg_pool=True,
        drop_rate=0.1,
        dropblock_size=5,
        is_flatten=True,
        maxpool_last2=True,
    ):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(
            blocks[0], planes[0], stride=2, drop_rate=drop_rate
        )
        self.layer2 = self._make_layer(
            blocks[1], planes[1], stride=2, drop_rate=drop_rate
        )

        self.layer3 = self._make_layer(
            blocks[2],
            planes[2],
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
            use_pool=maxpool_last2,
        )
        self.layer4 = self._make_layer(
            blocks[3],
            planes[3],
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
            use_pool=maxpool_last2,
        )
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.is_flatten = is_flatten
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        stride=1,
        drop_rate=0.0,
        drop_block=False,
        block_size=1,
        use_pool=True,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                drop_rate,
                drop_block,
                block_size,
                use_pool=use_pool,
            )
        )
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


def resnet12(
    keep_prob=1.0, avg_pool=True, is_flatten=True, maxpool_last2=True, **kwargs
):
    """Constructs a ResNet-12 model."""
    model = ResNet(
        [BasicBlock, BasicBlock, BasicBlock, BasicBlock],
        keep_prob=keep_prob,
        avg_pool=avg_pool,
        is_flatten=is_flatten,
        maxpool_last2=maxpool_last2,
        **kwargs
    )
    return model


def resnet12woLSC(
    keep_prob=1.0, avg_pool=True, is_flatten=True, maxpool_last2=True, **kwargs
):
    """Constructs a ResNet-12 model."""
    model = ResNet(
        [BasicBlock, BasicBlock, BasicBlock, BasicBlockWithoutResidual],
        planes=[64, 128, 256, 512],
        keep_prob=keep_prob,
        avg_pool=avg_pool,
        is_flatten=is_flatten,
        maxpool_last2=maxpool_last2,
        **kwargs
    )
    return model


if __name__ == "__main__":
    model = resnet12(avg_pool=True).cuda()
    data = torch.rand(10, 3, 84, 84).cuda()
    output = model(data)
    print(output.size())
