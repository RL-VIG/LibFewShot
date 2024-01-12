#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import weight_init

# Series informed activation function. Implemented by conv.
class activation(nn.ReLU):
    def __init__(self, dim, act_num=3):
        super(activation, self).__init__()
        self.act_num = act_num

        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))

        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):

        return self.bn(torch.nn.functional.conv2d(
            super(activation, self).forward(x),
            self.weight, padding=self.act_num, groups=self.dim))


class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2,  ada_pool=None):
        super().__init__()
        self.act_learn = 1
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim, eps=1e-6),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out, eps=1e-6)
        )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))

        self.act = activation(dim_out, act_num)
 
    def forward(self, x):

        x = self.conv1(x)
        
        # We use leakyrelu to implement the deep training technique.
        x = torch.nn.functional.leaky_relu(x,self.act_learn)
        
        x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
        return x

class VanillaNet(nn.Module):
    def __init__(self, in_chans=3, dims=[96, 192, 384, 768], 
                  act_num=3, strides=[2,2,2,1], ada_pool=None, 
                  is_feature=False, avg_pool=False, is_flatten=False, **kwargs):
        super().__init__()

        stride, padding = (4, 0) if not ada_pool else (3, 1)

        self.stem1 = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=stride, padding=padding),
            nn.BatchNorm2d(dims[0], eps=1e-6),
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(dims[0], eps=1e-6),
            activation(dims[0], act_num)
        )

        self.act_learn = 1

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i])
            else:
                stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], ada_pool=ada_pool[i])
            self.stages.append(stage)
        self.depth = len(strides)


        # self.cls1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Dropout(drop_rate),
        #     nn.Conv2d(dims[-1], num_classes, 1),
        #     nn.BatchNorm2d(num_classes, eps=1e-6),
        # )
        # self.cls2 = nn.Sequential(
        #     nn.Conv2d(num_classes, num_classes, 1)
        # )
        
        self.apply(self._init_weights)

        self.is_feature = is_feature
        self.avg_pool = avg_pool
        self.is_flatten = is_flatten
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        self.act_learn = m

    def forward(self, x):
        x = self.stem1(x)
        x = torch.nn.functional.leaky_relu(x,self.act_learn)
        x = self.stem2(x)

        for i in range(self.depth):
            x = self.stages[i](x)

        # x = self.cls1(x)
        # x = torch.nn.functional.leaky_relu(x,self.act_learn)
        # x = self.cls2(x)
        
        if self.avg_pool:
            x = self.avgpool(x)

        if self.is_flatten:
            x = x.view(x.size(0), -1)

        # if self.is_feature:
        #     return out1, out2, out3, out4

        return x



def vanillanet_6(**kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4, 1024*4], **kwargs)
    return model


if __name__ == "__main__":
    import torch

    model = vanillanet_6(is_flatten=True, avg_pool=True, strides=[2,2,2,1]).cuda()
    data = torch.rand(10, 3, 84, 84).cuda()
    output = model(data)
    print(output.size())



