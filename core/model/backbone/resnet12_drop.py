# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (
                    batch_size,
                    channels,
                    height - (self.block_size - 1),
                    width - (self.block_size - 1),
                )
            ).cuda()
            # print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            # print (block_mask.size())
            # print (x.size())
            countM = (
                block_mask.size()[0]
                * block_mask.size()[1]
                * block_mask.size()[2]
                * block_mask.size()[3]
            )
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = (
            torch.stack(
                [
                    torch.arange(self.block_size)
                    .view(-1, 1)
                    .expand(self.block_size, self.block_size)
                    .reshape(-1),  # - left_padding,
                    torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
                ]
            )
            .t()
            .cuda()
        )
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size**2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = (
                1.0
            )
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
        pool=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride, ceil_mode=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.register_buffer("num_batches_tracked", torch.tensor(0))
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.pool = pool

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
        if self.pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block is True:
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
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, drop_block=False, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        self.nFeat = 640
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
            drop_rate=drop_rate,
            drop_block=drop_block,
            block_size=dropblock_size,
        )
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
            drop_rate=drop_rate,
            drop_block=drop_block,
            block_size=dropblock_size,
            pool=False,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, pool=True
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, pool
            )
        )
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False, return_both=False, return_map=False):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # return [x1, x2, x3, x4]
        # return [x4]
        return x4
        # return [x3, x4]
        # return [x2, x3, x4]

    def forward_as_dict(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # result = self.fc(x)
        return x


def resnet12_drop(drop_block=True, **kwargs):
    """Constructs a ResNet-12 model."""
    model = ResNet(BasicBlock, drop_block=drop_block, **kwargs)
    return model
