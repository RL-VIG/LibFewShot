# -*- coding: utf-8 -*-
import torch


def mixup_data(x, y, _lambda):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = _lambda * x + (1 - _lambda) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, _lambda
