# -*- coding: utf-8 -*-
"""
@article{DBLP:journals/corr/abs-1812-03664,
  author    = {Han{-}Jia Ye and
               Hexiang Hu and
               De{-}Chuan Zhan and
               Fei Sha},
  title     = {Learning Embedding Adaptation for Few-Shot Learning},
  year      = {2018},
  archivePrefix = {arXiv},
  eprint    = {1812.03664},
}
http://arxiv.org/abs/1812.03664

Adapted from https://github.com/Sha-Lab/FEAT.
"""

import torch
from torch import nn

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from ..metric.proto_net import ProtoLayer


class FEAT_Pretrain(FinetuningModel):
    def __init__(
        self, feat_dim, train_num_class, val_num_class, mode="euclidean", **kwargs
    ):
        super(FEAT_Pretrain, self).__init__(**kwargs)
        self.train_num_class = train_num_class
        self.val_num_class = val_num_class
        self.feat_dim = feat_dim

        self.train_classifier = nn.Linear(self.feat_dim, self.train_num_class)
        self.val_classifier = ProtoLayer()
        self.mode = mode
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        # FIXME:  do not do validation in first 500 epoches # # test on 16-way 1-shot
        """
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output = self.val_classifier(
            query_feat,
            support_feat,
            self.way_num,
            self.shot_num,
            self.query_num,
            mode=self.mode,
        ).reshape(-1, self.way_num)

        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.train_classifier(feat)

        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        raise NotImplementedError
