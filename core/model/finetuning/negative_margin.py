# -*- coding: utf-8 -*-
"""
@article{liu2020negative,
  title={Negative Margin Matters: Understanding Margin in Few-shot Classification},
  author={Liu, Bin and Cao, Yue and Lin, Yutong and Li, Qi and Zhang, Zheng and Long, Mingsheng and Hu, Han},
  journal={arXiv preprint arXiv:2003.12060},
  year={2020}
}
"""
import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from torch.nn import Parameter
from .finetuning_model import FinetuningModel
import math
from torch.optim.lr_scheduler import _LRScheduler


class NegLayer(nn.Module):
    def __init__(self, in_features, out_features, margin=0.40, scale_factor=30.0):
        super(NegLayer, self).__init__()
        self.margin = margin
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        # when test, no label, just return
        if label is None:
            return cosine * self.scale_factor

        phi = cosine - self.margin

        output = torch.where(self.one_hot(label, cosine.shape[1]).byte(), phi, cosine)
        output *= self.scale_factor
        return output

    def one_hot(self, y, num_class):
        return (
            torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)
        )


class NegNet(FinetuningModel):
    def __init__(self, feat_dim, num_class, margin=-0.3, scale_factor=30.0, **kwargs):
        super(NegNet, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.margin = margin
        self.scale_factor = scale_factor
        self.NegLayer = NegLayer(feat_dim, num_class, margin, scale_factor)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)
        # support_target = support_target.reshape(episode_size, self.way_num*self.shot_num)

        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(
                support_feat[i], support_target[i], query_feat[i]
            )
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        classifier = NegLayer(
            self.feat_dim,
            self.test_way,
            self.inner_param["inner_margin"],
            self.inner_param["inner_scale_factor"],
        )
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)
        classifier.train()

        loss_func = nn.CrossEntropyLoss()

        support_size = support_feat.size(0)
        batch_size = 4
        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, batch_size):
                select_id = rand_id[i : min(i + batch_size, support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]
                # print("target:")
                # print(target)
                output = classifier(batch, target)

                loss = loss_func(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)  # retain_graph = True
                optimizer.step()

        output = classifier(query_feat)
        return output

    def set_forward_loss(self, batch):
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        feat = self.emb_func(image)
        output = self.NegLayer(feat, target.reshape(-1))

        loss = self.loss_func(output, target.reshape(-1))
        acc = accuracy(output, target.reshape(-1))
        return output, acc, loss
