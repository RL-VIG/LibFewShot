# -*- coding: utf-8 -*-
"""
FGFL-specific ResNet-12 implementation.
Based on the original FGFL ResNet with modifications for LibFewShot
integration.

This ResNet network was designed following the practice of the following
papers:
TADAM: Task dependent adaptive metric for improved few-shot learning
(Oreshkin et al., in NIPS 2018) and A Simple Neural Attentive Meta-Learner
(Mishra et al., in ICLR 2018).

Key differences from standard ResNet-12:
1. Returns both feature map and flattened features: (m, x)
2. Feature map (m) is used for FGFL's hook mechanism and frequency mask
   generation
3. Maintains compatibility with FGFL's frequency domain processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.backbone.utils.dropblock import DropBlock


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class FGFLBasicBlock(nn.Module):
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
    ):
        super(FGFLBasicBlock, self).__init__()
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


class FGFLResNet(nn.Module):

    def __init__(
        self,
        block=FGFLBasicBlock,
        keep_prob=1.0,
        avg_pool=True,
        drop_rate=0.1,
        dropblock_size=5,
    ):
        self.inplanes = 3
        super(FGFLResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
        )
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=dropblock_size,
        )
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1
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
            )
        )
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        m = self.layer4(x)

        # m = dct.dct_2d(m, norm='ortho')

        if self.keep_avg_pool:
            x = self.avgpool(m)
        x = x.view(x.size(0), -1)
        # print(m.shape)
        # print(x.shape)
        return m, x
        # return x


def fgfl_resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """
    Constructs a FGFL-specific ResNet-12 model.

    Args:
        keep_prob (float): Keep probability for dropout. Default: 1.0
        avg_pool (bool): Whether to use average pooling. Default: False
                        (FGFL typically uses avg_pool=False)
        **kwargs: Additional arguments

    Returns:
        FGFLResNet: FGFL-specific ResNet-12 model that returns
                   (feature_map, features)
    """
    model = FGFLResNet(FGFLBasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


if __name__ == "__main__":
    # Test the FGFL ResNet
    model = fgfl_resnet12(avg_pool=True)
    if torch.cuda.is_available():
        model = model.cuda()
        data = torch.rand(10, 3, 84, 84).cuda()
    else:
        data = torch.rand(10, 3, 84, 84)

    feature_map, features = model(data)
    print(f"Feature map shape: {feature_map.size()}")
    print(f"Flattened features shape: {features.size()}")
    print("FGFL ResNet-12 test passed!")
