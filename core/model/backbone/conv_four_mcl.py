# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Conv64F_MCL(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(Conv64F_MCL, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.out_channels = 64

        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.xavier_uniform_(l.weight)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)
            elif isinstance(l, nn.Linear):
                torch.nn.init.normal_(l.weight, 0, 0.01)
                if l.bias is not None:
                    torch.nn.init.constant(l.bias, 0)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
