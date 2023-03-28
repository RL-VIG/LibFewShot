# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/nips/HouCMSC19,
  author    = {Ruibing Hou and
               Hong Chang and
               Bingpeng Ma and
               Shiguang Shan and
               Xilin Chen},
  title     = {Cross Attention Network for Few-shot Classification},
  booktitle = {Advances in Neural Information Processing Systems 32: Annual Conference
               on Neural Information Processing Systems 2019, NeurIPS 2019, December
               8-14, 2019, Vancouver, BC, Canada},
  pages     = {4005--4016},
  year      = {2019},
  url       = {https://proceedings.neurips.cc/paper/2019/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html}
}
https://arxiv.org/abs/1910.07677

Adapted from https://github.com/blue-blue272/fewshot-CAN.
"""
from __future__ import absolute_import
from __future__ import division
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from core.model.metric.metric_model import MetricModel


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        loss = (-targets * log_probs).mean(0).sum()
        return loss / inputs.size(2)


def one_hot(indices, depth, use_cuda=True):
    if use_cuda:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    else:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies


def shuffle(images, targets, global_targets):
    """
    A trick for CAN training
    """
    sample_num = images.shape[1]
    for i in range(4):
        indices = torch.randperm(sample_num).to(images.device)
        images = images.index_select(1, indices)
        targets = targets.index_select(1, indices)
        global_targets = global_targets.index_select(1, indices)
    return images, targets, global_targets


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.
    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    """
    Support & Query share one attention
    """

    def __init__(self, mid_channels):
        super(CAM, self).__init__()
        self.conv1 = ConvBlock(mid_channels * mid_channels, mid_channels, 1)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels * mid_channels, 1, stride=1, padding=0
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def get_attention(self, a):
        input_a = a
        # print('line ', sys._getframe().f_lineno, a.shape)
        a = a.mean(3)  # GAP
        # print('line ', sys._getframe().f_lineno, a.shape)
        a = a.transpose(1, 3)
        # print('line ', sys._getframe().f_lineno, a.shape)
        a = F.relu(self.conv1(a))
        # print('line ', sys._getframe().f_lineno, a.shape)
        a = self.conv2(a)
        # print('line ', sys._getframe().f_lineno, a.shape)
        a = a.transpose(1, 3)
        # print('line ', sys._getframe().f_lineno, a.shape)
        a = a.unsqueeze(3)
        # print('line ', sys._getframe().f_lineno, a.shape)

        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a

    def forward(self, f1, f2):
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)

        # Flatten
        f1 = f1.reshape(b, n1, c, -1)
        f2 = f2.reshape(b, n2, c, -1)

        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)  # [1, 5, 512, 36]
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)  # [1, 75, 512, 36]
        # print('line ', sys._getframe().f_lineno, f1_norm.shape, f2_norm.shape)
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
        f2_norm = f2_norm.unsqueeze(1)

        a1 = torch.matmul(f1_norm, f2_norm)  # [1, 5, 75, 36, 36]
        a2 = a1.transpose(3, 4)  # [1, 5, 75, 36, 36]
        # print('line ', sys._getframe().f_lineno, a1.shape, a2.shape)
        a1 = self.get_attention(a1)  # [1, 5, 75, 36]
        a2 = self.get_attention(a2)  # [1, 5, 75, 36]

        f1 = f1.unsqueeze(2) * a1.unsqueeze(3)
        f1 = f1.reshape(b, n1, n2, c, h, w)
        f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
        f2 = f2.reshape(b, n1, n2, c, h, w)
        return f1.transpose(1, 2), f2.transpose(1, 2)


class CAMLayer(nn.Module):
    def __init__(
        self, scale_cls, iter_num_prob=35.0 / 75, num_classes=64, nFeat=512, HW=5
    ):
        super(CAMLayer, self).__init__()
        self.scale_cls = scale_cls
        self.cam = CAM(HW)
        self.iter_num_prob = iter_num_prob
        self.nFeat = nFeat
        self.classifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)

    def val(self, support_feat, query_feat):
        query_feat = query_feat.mean(4)
        query_feat = query_feat.mean(4)
        support_feat = F.normalize(
            support_feat, p=2, dim=support_feat.dim() - 1, eps=1e-12
        )  # [1, 75, 5, 512]
        query_feat = F.normalize(
            query_feat, p=2, dim=query_feat.dim() - 1, eps=1e-12
        )  # [1, 75, 5, 512]
        scores = self.scale_cls * torch.sum(
            query_feat * support_feat, dim=-1
        )  # [1, 75, 5]
        return scores

    def forward(self, support_feat, query_feat, support_targets, query_targets):
        """
        support_feat: [4, 5, 512, 6, 6]
        query_feat: [4, 75, 512, 6, 6]
        support_targets: [4, 5, 5] one-hot
        query_targets: [4, 75, 5] one-hot
        """
        original_feat_shape = support_feat.size()
        batch_size = support_feat.size(0)
        n_support = support_feat.size(1)
        n_query = query_feat.size(1)
        way_num = support_targets.size(-1)
        # feat = support_feat.size(-1)

        # flatten feature
        support_feat = support_feat.reshape(batch_size, n_support, -1)

        labels_train_transposed = support_targets.transpose(1, 2)  # [1, 5, 5]

        # calc the prototypes of support set
        prototypes = torch.bmm(
            labels_train_transposed, support_feat
        )  # [1, 5, 5]x[1, 5, 640]
        prototypes = prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )  # [1, 5, 640]
        prototypes = prototypes.reshape(
            batch_size, -1, *original_feat_shape[2:]
        )  # [1, 5, 512, 6, 6]
        prototypes, query_feat = self.cam(
            prototypes, query_feat
        )  # [1, 75, 5, 512, 6, 6]  # [2, 75, 640, 1, 1]
        prototypes = prototypes.mean(4)
        prototypes = prototypes.mean(4)  # [1, 75, 5, 512]

        if not self.training:
            return self.val(prototypes, query_feat)

        proto_norm = F.normalize(prototypes, p=2, dim=3, eps=1e-12)
        query_norm = F.normalize(query_feat, p=2, dim=3, eps=1e-12)
        proto_norm = proto_norm.unsqueeze(4)
        proto_norm = proto_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(query_norm * proto_norm, dim=3)
        cls_scores = cls_scores.reshape(batch_size * n_query, *cls_scores.size()[2:])

        query_feat = query_feat.reshape(batch_size, n_query, way_num, -1)
        query_feat = query_feat.transpose(2, 3)
        query_targets = query_targets.unsqueeze(3)
        query_feat = torch.matmul(query_feat, query_targets)
        query_feat = query_feat.reshape(
            batch_size * n_query, -1, *original_feat_shape[-2:]
        )
        query_targets = self.classifier(query_feat)

        return query_targets, cls_scores

    def helper(self, support_feat, query_feat, support_targets):
        """
        support_targets_transposed: one-hot
        """
        b, n, c, h, w = support_feat.size()

        support_targets_transposed = support_targets.transpose(1, 2)
        support_feat = torch.bmm(
            support_targets_transposed, support_feat.reshape(b, n, -1)
        )
        support_feat = support_feat.div(
            support_targets_transposed.sum(dim=2, keepdim=True).expand_as(support_feat)
        )
        support_feat = support_feat.reshape(b, -1, c, h, w)

        support_feat, query_feat = self.cam(support_feat, query_feat)
        support_feat = support_feat.mean(-1).mean(-1)
        query_feat = query_feat.mean(-1).mean(-1)

        query_feat = F.normalize(query_feat, p=2, dim=query_feat.dim() - 1, eps=1e-12)
        support_feat = F.normalize(
            support_feat, p=2, dim=support_feat.dim() - 1, eps=1e-12
        )
        scores = self.scale_cls * torch.sum(query_feat * support_feat, dim=-1)
        return scores

    # def val_transductive(self, support_feat, query_feat, support_targets, query_targets):
    #     iter_num_prob = self.iter_num_prob
    #     batch_size, num_train = support_feat.size(0), support_feat.size(1)
    #     num_test = query_feat.size(1)
    #     K = support_targets.size(2)
    #
    #     cls_scores = self.helper(support_feat, query_feat, support_targets)
    #
    #     num_images_per_iter = int(num_test * iter_num_prob)
    #     num_iter = num_test // num_images_per_iter
    #
    #     for i in range(num_iter):
    #         max_scores, preds = torch.max(cls_scores, 2)
    #         chose_index = torch.argsort(max_scores.view(-1), descending=True)
    #         chose_index = chose_index[: num_images_per_iter * (i + 1)]
    #         ftest_iter = query_feat[0, chose_index].unsqueeze(0)
    #         preds_iter = preds[0, chose_index].unsqueeze(0)
    #
    #         preds_iter = one_hot(preds_iter.view(-1), K).cuda()
    #         preds_iter = preds_iter.view(batch_size, -1, K)
    #
    #         support_feat_iter = torch.cat((support_feat, ftest_iter), 1)
    #         support_targets_iter = torch.cat((support_targets, preds_iter), 1)
    #         cls_scores = self.helper(support_feat_iter, query_feat, support_targets_iter)
    #
    #     return cls_scores


class CAN(MetricModel):
    def __init__(
        self,
        scale_cls,
        iter_num_prob=35.0 / 75,
        num_classes=64,
        nFeat=512,
        HW=5,
        **kwargs
    ):
        super(CAN, self).__init__(**kwargs)
        self.cam_layer = CAMLayer(scale_cls, iter_num_prob, num_classes, nFeat, HW)
        self.loss_func = CrossEntropyLoss()
        self._init_network()

    def set_forward(
        self,
        batch,
    ):
        """
        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        global_targets = global_targets.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        emb = self.emb_func(images)
        (
            support_feat,
            query_feat,
            support_targets,
            query_targets,
        ) = self.split_by_episode(emb, mode=2)

        # convert to one-hot
        support_targets_one_hot = one_hot(
            support_targets.reshape(episode_size * self.way_num * self.shot_num),
            self.way_num,
        )
        support_targets_one_hot = support_targets_one_hot.reshape(
            episode_size, self.way_num * self.shot_num, self.way_num
        )
        query_targets_one_hot = one_hot(
            query_targets.reshape(episode_size * self.way_num * self.query_num),
            self.way_num,
        )
        query_targets_one_hot = query_targets_one_hot.reshape(
            episode_size, self.way_num * self.query_num, self.way_num
        )
        cls_scores = self.cam_layer(
            support_feat, query_feat, support_targets_one_hot, query_targets_one_hot
        )
        # cls_scores = self.cam_layer.val_transductive(
        #        support_feat, query_feat, support_targets_one_hot, query_targets_one_hot
        # )

        cls_scores = cls_scores.reshape(
            episode_size * self.way_num * self.query_num, -1
        )
        acc = accuracy(cls_scores, query_targets.reshape(-1), topk=1)
        return cls_scores, acc

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        global_targets = global_targets.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        emb = self.emb_func(images)  # [80, 640]
        (
            support_feat,
            query_feat,
            support_targets,
            query_targets,
        ) = self.split_by_episode(
            emb, mode=2
        )  # [4,5,512,6,6] [4,
        # 75, 512,6,6] [4, 5] [300]
        support_targets = support_targets.reshape(
            episode_size, self.way_num * self.shot_num
        ).contiguous()
        support_global_targets, query_global_targets = (
            global_targets[:, :, : self.shot_num],
            global_targets[:, :, self.shot_num :],
        )

        support_feat, support_targets, support_global_targets = shuffle(
            support_feat,
            support_targets,
            support_global_targets.reshape(*support_targets.size()),
        )
        query_feat, query_targets, query_global_targets = shuffle(
            query_feat,
            query_targets.reshape(*query_feat.size()[:2]),
            query_global_targets.reshape(*query_feat.size()[:2]),
        )

        # convert to one-hot
        support_targets_one_hot = one_hot(
            support_targets.reshape(episode_size * self.way_num * self.shot_num),
            self.way_num,
        )
        support_targets_one_hot = support_targets_one_hot.reshape(
            episode_size, self.way_num * self.shot_num, self.way_num
        )
        query_targets_one_hot = one_hot(
            query_targets.reshape(episode_size * self.way_num * self.query_num),
            self.way_num,
        )
        query_targets_one_hot = query_targets_one_hot.reshape(
            episode_size, self.way_num * self.query_num, self.way_num
        )
        # print(support_feat.shape, query_feat.shape, support_targets_one_hot.shape, query_targets_one_hot.shape)
        # [75, 64, 6, 6], [75, 5, 6, 6]
        output, cls_scores = self.cam_layer(
            support_feat, query_feat, support_targets_one_hot, query_targets_one_hot
        )
        loss1 = self.loss_func(output, query_global_targets.contiguous().reshape(-1))
        loss2 = self.loss_func(cls_scores, query_targets.reshape(-1))
        loss = loss1 + 0.5 * loss2
        cls_scores = torch.sum(
            cls_scores.reshape(*cls_scores.size()[:2], -1), dim=-1
        )  # [300, 5]
        acc = accuracy(cls_scores, query_targets.reshape(-1), topk=1)
        return output, acc, loss
