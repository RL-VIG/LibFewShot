# -*- coding: utf-8 -*-
from abc import abstractmethod
import einops

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from core.utils import ModelType
from .init import init_weights


class AbstractModel(nn.Module):
    def __init__(self, init_type, model_type=ModelType.ABSTRACT, **kwargs):
        super(AbstractModel, self).__init__()

        self.init_type = init_type
        self.model_type = model_type
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    def forward(self, x):
        if self.training:
            return self.set_forward_loss(x)
        else:
            return self.set_forward(x)

    def train(self, mode=True):
        super(AbstractModel, self).train(mode)
        # for methods with distiller
        if hasattr(self, "distill_layer"):
            self.distill_layer.train(False)

    def eval(self):
        return super(AbstractModel, self).eval()

    def _init_network(self):
        init_weights(self, self.init_type)

    def _generate_local_targets(self, episode_size):
        local_targets = (
            torch.arange(self.way_num, dtype=torch.long)
            .view(1, -1, 1)
            .repeat(episode_size, 1, self.shot_num + self.query_num)
            .view(-1)
        )
        return local_targets

    def split_by_episode(self, features, mode):
        """
        split features by episode and
        generate local targets + split labels by episode
        """
        episode_size = features.size(0) // (self.way_num * (self.shot_num + self.query_num))
        local_labels = (
            self._generate_local_targets(episode_size)
            .to(self.device)
            .contiguous()
            .view(episode_size, self.way_num, self.shot_num + self.query_num)
        )

        if mode == 1:  # input 2D, return 3D(with episode) E.g.ANIL & R2D2
            features = features.contiguous().view(
                episode_size, self.way_num, self.shot_num + self.query_num, -1
            )
            support_features = (
                features[:, :, : self.shot_num, :]
                .contiguous()
                .view(episode_size, self.way_num * self.shot_num, -1)
            )
            query_features = (
                features[:, :, self.shot_num :, :]
                .contiguous()
                .view(episode_size, self.way_num * self.query_num, -1)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
        elif mode == 2:  # input 4D, return 5D(with episode) E.g.DN4
            b, c, h, w = features.shape
            features = features.contiguous().view(
                episode_size,
                self.way_num,
                self.shot_num + self.query_num,
                c,
                h,
                w,
            )
            support_features = (
                features[:, :, : self.shot_num, :, ...]
                .contiguous()
                .view(episode_size, self.way_num * self.shot_num, c, h, w)
            )
            query_features = (
                features[:, :, self.shot_num :, :, ...]
                .contiguous()
                .view(episode_size, self.way_num * self.query_num, c, h, w)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
        elif mode == 3:  # input 4D, return 4D(w/o episode) E.g.realationnet
            b, c, h, w = features.shape
            features = features.contiguous().view(
                self.way_num, self.shot_num + self.query_num, c, h, w
            )
            support_features = (
                features[:, : self.shot_num, :, ...]
                .contiguous()
                .view(self.way_num * self.shot_num, c, h, w)
            )
            query_features = (
                features[:, self.shot_num :, :, ...]
                .contiguous()
                .view(self.way_num * self.query_num, c, h, w)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
        elif (
            mode == 4
        ):  # finetuning baseline input 2D, return 2D(w/o episode) E.g.baseline set_forward
            features = features.view(self.way_num, self.shot_num + self.query_num, -1)
            support_features = (
                features[:, : self.shot_num, :].contiguous().view(self.way_num * self.shot_num, -1)
            )
            query_features = (
                features[:, self.shot_num :, :].contiguous().view(self.way_num * self.query_num, -1)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                self.way_num * self.query_num
            )
        else:
            raise Exception("mode should in [1,2,3,4], not {}".format(mode))

        return support_features, query_features, support_target, query_target

    def reverse_setting_info(self):
        (
            self.way_num,
            self.shot_num,
            self.query_num,
            self.test_way,
            self.test_shot,
            self.test_query,
        ) = (
            self.test_way,
            self.test_shot,
            self.test_query,
            self.way_num,
            self.shot_num,
            self.query_num,
        )

    def cal_loss(self, output, target, reduction="mean", **kwargs):
        # CE loss
        try:
            if self.loss["name"] == "CrossEntropyLoss":
                if output.dim() >= 3:
                    # [episode_size, way_num * query_num, way_num]
                    output = output.reshape(-1, output.size(-1))
                if target.dim() >= 2:
                    # [episode_size, way_num * query_num]
                    target = target.reshape(-1)
                assert reduction in ["mean", "sum"]
                return F.cross_entropy(output, target, reduction=reduction)
            elif self.loss["name"] == "SoftlabelCrossEntropyLoss":
                if output.dim() >= 3:
                    # [episode_size, way_num * query_num, way_num]
                    output = output.reshape(-1, output.size(-1))
                if target.dim() >= 3:
                    # [episode_size, way_num * query_num, way_num]
                    target = target.reshape(-1, target.size(-1))
                log_prob = -F.log_softmax(output, dim=1)
                nll_loss = torch.sum(torch.mul(log_prob, target), dim=1)
                assert reduction in ["mean", "sum"]
                if reduction == "mean":
                    nll_loss = nll_loss.mean()
                else:
                    nll_loss = nll_loss.sum()
                return nll_loss
            elif self.loss["name"] == "LabelSmoothCrossEntropyLoss":
                if "num_classes" not in kwargs:
                    num_classes = self.way_num
                else:
                    num_classes = kwargs["num_classes"]
                target = F.one_hot(target, num_classes)
                log_prob = F.log_softmax(output, dim=-1)
                nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_prob.mean(dim=-1)
                loss = (1.0 - self.loss["kwargs"]["smoothing"]) * nll_loss + self.loss["kwargs"]["smoothing"] * smooth_loss
                return loss.mean()
            elif self.loss["name"] == "KLDivergenceLoss":
                if "T" not in self.loss["kwargs"]:
                    T = 1.0
                else:
                    T = self.loss["kwargs"]["T"]
                log_prob = F.log_softmax(output / T, dim=1)
                log_targ = F.softmax(target / T, dim=1)
                return F.kl_div(log_prob, log_targ, size_average=False) * (T ** 2).mean()
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            print("Please confirm your loss config, output shape and target shape")