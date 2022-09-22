# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/icml/FinnAL17,
  author    = {Chelsea Finn and
               Pieter Abbeel and
               Sergey Levine},
  title     = {Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning,
               {ICML} 2017, Sydney, NSW, Australia, 6-11 August 2017},
  series    = {Proceedings of Machine Learning Research},
  volume    = {70},
  pages     = {1126--1135},
  publisher = {{PMLR}},
  year      = {2017},
  url       = {http://proceedings.mlr.press/v70/finn17a.html}
}
https://arxiv.org/abs/1703.03400

Adapted from https://github.com/wyharveychen/CloserLookFewShot.
"""
import torch
from torch import nn

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module


class MAMLLayer(nn.Module):
    def __init__(self, feat_dim=64, way_num=5) -> None:
        super(MAMLLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)


class MAML(MetaModel):
    def __init__(self, inner_param, feat_dim, **kwargs):
        super(MAML, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
        self.inner_param = inner_param

        convert_maml_module(self)

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def set_forward(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_targets = query_targets[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, support_target):
        lr = self.inner_param["lr"]
        fast_parameters = list(self.parameters())
        for parameter in self.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()
        for i in range(
            self.inner_param["train_iter"]
            if self.training
            else self.inner_param["test_iter"]
        ):
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - lr * grad[k]
                fast_parameters.append(weight.fast)
