# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/OhYKY21,
  author    = {Jaehoon Oh and
               Hyungjun Yoo and
               ChangHwan Kim and
               Se{-}Young Yun},
  title     = {{BOIL:} Towards Representation Change for Few-shot Learning},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=umIdUL8rMH},
}
https://arxiv.org/abs/2008.08882

Adapted from https://github.com/HJ-Yoo/BOIL.
"""
import torch
from torch import nn

from core.utils import accuracy
from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module


class BOILLayer(nn.Module):
    def __init__(self, feat_dim=64, way_num=5) -> None:
        super(BOILLayer, self).__init__()
        self.layers = nn.Sequential(nn.Linear(feat_dim, way_num))

    def forward(self, x):
        return self.layers(x)


class BOIL(MetaModel):
    def __init__(self, inner_param, feat_dim, testing_method, **kwargs):
        super(BOIL, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = BOILLayer(feat_dim, way_num=self.way_num)
        self.inner_param = inner_param
        self.testing_method = testing_method

        convert_maml_module(self)

    def forward_output(self, x):
        feat_wo_head = self.emb_func(x)
        feat_w_head = self.classifier(feat_wo_head)
        return feat_wo_head, feat_w_head

    def set_forward(self, batch):
        image, global_target = batch  # unused global_target
        image, global_target = batch
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
            if self.testing_method == "Directly":
                _, output = self.forward_output(episode_query_image)
            elif self.testing_method == "Once_update":
                self.set_forward_adaptation(
                    episode_support_image, episode_support_target
                )
                _, output = self.forward_output(episode_query_image)
            elif self.testing_method == "NIL":
                support_features, _ = self.forward_output(episode_support_image)
                query_features, _ = self.forward_output(episode_query_image)
                support_features_mean = torch.mean(
                    support_features.reshape(self.way_num, self.shot_num, -1), dim=1
                )
                output = nn.CosineSimilarity()(
                    query_features.unsqueeze(-1),
                    support_features_mean.transpose(-1, -2).unsqueeze(0),
                )
            else:
                raise NotImplementedError(
                    'Unknown testing method. The testing_method should in ["NIL", "Directly","Once_update"]'
                )

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

            features, output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1)) / episode_size
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, support_target):
        extractor_lr = self.inner_param["extractor_lr"]
        classifier_lr = self.inner_param["classifier_lr"]
        fast_parameters = list(item[1] for item in self.named_parameters())
        for parameter in self.parameters():
            parameter.fast = None
        self.emb_func.train()
        self.classifier.train()
        features, output = self.forward_output(support_set)
        loss = self.loss_func(output, support_target)
        grad = torch.autograd.grad(
            loss, fast_parameters, create_graph=True, allow_unused=True
        )
        fast_parameters = []

        for k, weight in enumerate(self.named_parameters()):
            if grad[k] is None:
                continue
            lr = classifier_lr if "Linear" in weight[0] else extractor_lr
            if weight[1].fast is None:
                weight[1].fast = weight[1] - lr * grad[k]
            else:
                weight[1].fast = weight[1].fast - lr * grad[k]
            fast_parameters.append(weight[1].fast)
