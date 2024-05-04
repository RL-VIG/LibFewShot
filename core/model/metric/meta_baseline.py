# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class ProtoLayer_temperature(nn.Module):
    def __init__(self):
        super(ProtoLayer_temperature, self).__init__()

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        query_num,
        mode="cos_sim"
        ):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # t, wq, c
        query_feat = query_feat.reshape(t, way_num * query_num, c)
        # t, w, c
        support_feat = support_feat.reshape(t, way_num, shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)
        return {
            # t, wq, 1, c - t, 1, w, c -> t, wq, w
            "euclidean": lambda x, y: -torch.sum(
                torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
                dim=3,
            )
            ,
            # t, wq, c - t, c, w -> t, wq, w
            "cos_sim": lambda x, y: torch.matmul(
                F.normalize(x, p=2, dim=-1),
                torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # FEAT did not normalize the query_feat
            )
            ,
        }[mode](query_feat, proto_feat)


class MetaBaseline(MetricModel):
    def __init__(self, **kwargs):
        super(MetaBaseline, self).__init__(**kwargs)
        self.proto_layer = ProtoLayer_temperature()
        self.loss_func = nn.CrossEntropyLoss()
        self.temp=nn.Parameter(torch.tensor(10.))
        # self.temp=torch.tensor(10.)
    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)*self.temp
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        emb = self.emb_func(images)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            emb, mode=1
        )

        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)*self.temp
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss
