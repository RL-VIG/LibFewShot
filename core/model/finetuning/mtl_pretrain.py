# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/SunLCS19,
  author    = {Qianru Sun and
               Yaoyao Liu and
               Tat{-}Seng Chua and
               Bernt Schiele},
  title     = {Meta-Transfer Learning for Few-Shot Learning},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}
               2019, Long Beach, CA, USA, June 16-20, 2019},
  pages     = {403--412},
  year      = {2019},
  url       = {http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Meta-Transfer_Learning_for_Few
  -Shot_Learning_CVPR_2019_paper.html},
  doi       = {10.1109/CVPR.2019.00049}
}
https://arxiv.org/abs/1812.02391

Adapted from https://github.com/yaoyao-liu/meta-transfer-learning.
"""
import torch
from torch import nn

from core.utils import accuracy
from .finetuning_model import FinetuningModel
import torch.nn.functional as F


# FIXME: Add multi-GPU support
class MTLBaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, way_num, z_dim):
        super().__init__()
        self.way_num = way_num
        self.z_dim = z_dim
        self.var = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.way_num, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.var.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.way_num))
        self.var.append(self.fc1_b)

    def forward(self, input_x, the_var=None):
        if the_var is None:
            the_var = self.var
        fc1_w = the_var[0]
        fc1_b = the_var[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.var


class MTLPretrain(FinetuningModel):  # use image-size=80 in repo
    def __init__(self, feat_dim, num_classes, inner_param, **kwargs):
        super(MTLPretrain, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.pre_fc = nn.Sequential(
            nn.Linear(self.feat_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.num_classes),
        )
        self.base_learner = MTLBaseLearner(self.way_num, z_dim=self.feat_dim)
        self.inner_param = inner_param

        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """
        meta-validation
        :param batch:
        :return:
        """
        image, _ = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=4)

        classifier, fast_weight = self.set_forward_adaptation(support_feat, support_target)

        output = classifier(query_feat, fast_weight)

        acc = accuracy(output, query_target.contiguous().reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """
        finetuning
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device).contiguous()

        feat = self.emb_func(image)

        output = self.pre_fc(feat).contiguous()

        loss = self.loss_func(output, global_target)
        acc = accuracy(output, global_target)
        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = self.base_learner.to(self.device)

        logit = self.base_learner(support_feat)
        loss = self.loss_func(logit, support_target)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_parameters = list(
            map(
                lambda p: p[1] - 0.01 * p[0],
                zip(grad, self.base_learner.parameters()),
            )
        )

        for _ in range(1, self.inner_param["iter"]):
            logit = self.base_learner(support_feat, fast_parameters)
            loss = F.cross_entropy(logit, support_target)
            grad = torch.autograd.grad(loss, fast_parameters)
            fast_parameters = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_parameters)))

        return classifier, fast_parameters
