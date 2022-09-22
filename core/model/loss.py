# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import functional as F
import torch


class L2DistLoss(nn.Module):
    def __init__(self):
        super(L2DistLoss, self).__init__()

    def forward(self, feat1, feat2):
        loss = torch.mean(torch.sqrt(torch.sum((feat1 - feat2) ** 2, dim=1)))
        if torch.isnan(loss).any():
            loss = 0.0
        return loss


class LabelSmoothCELoss(nn.Module):
    def __init__(self, smoothing):
        super(LabelSmoothCELoss, self).__init__()

        self.smoothing = smoothing

    def forward(self, output, target):
        log_prob = F.log_softmax(output, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class DistillKLLoss(nn.Module):
    def __init__(self, T):
        super(DistillKLLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        if y_t is None:
            return 0.0

        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.size(0)
        return loss
