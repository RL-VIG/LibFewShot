# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/SungYZXTH18,
  author    = {Flood Sung and
               Yongxin Yang and
               Li Zhang and
               Tao Xiang and
               Philip H. S. Torr and
               Timothy M. Hospedales},
  title     = {Learning to Compare: Relation Network for Few-Shot Learning},
  booktitle = {2018 {IEEE} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2018, Salt Lake City, UT, USA, June 18-22, 2018},
  pages     = {1199--1208},
  year      = {2018},
  url       = {http://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html},
  doi       = {10.1109/CVPR.2018.00131}
}
https://arxiv.org/abs/1711.06025

Adapted from https://github.com/floodsung/LearningToCompare_FSL.
"""
import torch
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class RelationLayer(nn.Module):
    def __init__(self, feat_dim=64, feat_height=3, feat_width=3):
        super(RelationLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(feat_dim * feat_height * feat_width, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        # print(x.shape)
        out = self.layers(x)
        out = out.reshape(x.size(0), -1)
        out = self.fc(out)
        return out


class RelationNet(MetricModel):
    def __init__(self, feat_dim=64, feat_height=3, feat_width=3, **kwargs):
        super(RelationNet, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.feat_height = feat_height
        self.feat_width = feat_width
        self.relation_layer = RelationLayer(
            self.feat_dim, self.feat_height, self.feat_width
        )
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )

        relation_pair = self._calc_pairs(query_feat, support_feat)
        output = self.relation_layer(relation_pair).reshape(-1, self.way_num)

        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )

        relation_pair = self._calc_pairs(query_feat, support_feat)
        output = self.relation_layer(relation_pair).reshape(-1, self.way_num)

        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc, loss

    def _calc_pairs(self, query_feat, support_feat):
        """

        :param query_feat: (task_num, query_num * way_num, feat_dim, feat_width, feat_height)
        :param support_feat: (task_num, support_num * way_num, feat_dim, feat_width, feat_height)
        :return: query_num * way_num * way_num, feat_dim, feat_width, feat_height
        """
        t, _, c, h, w = query_feat.size()
        # t, w, wq, c, h, w -> t, wq, w, c, h, w
        query_feat = query_feat.unsqueeze(1).repeat(1, self.way_num, 1, 1, 1, 1)
        query_feat = torch.transpose(query_feat, 1, 2)

        # t, w, s, c, h, w -> t, 1, w, c, h, w -> t, wq, w, c, h, w
        support_feat = support_feat.reshape(t, self.way_num, self.shot_num, c, h, w)
        support_feat = (
            torch.sum(support_feat, dim=(2,))
            .unsqueeze(1)
            .repeat(1, self.way_num * self.query_num, 1, 1, 1, 1)
        )

        # t, wq, w, 2c, h, w -> twqw, 2c, h, w
        relation_pair = torch.cat((query_feat, support_feat), dim=3).reshape(
            -1, c * 2, h, w
        )
        return relation_pair
