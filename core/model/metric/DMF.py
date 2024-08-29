# -*- coding: utf-8 -*-
"""
@inproceedings{xu2021dmf,
  title={Learning Dynamic Alignment via Meta-filter for Few-shot Learning},
  author={Chengming Xu and Chen Liu and Li Zhang and Chengjie Wang and Jilin Li and Feiyue Huang and Xiangyang Xue and Yanwei Fu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
https://arxiv.org/pdf/2103.13582

Adapted from https://github.com/chmxu/Dynamic-Meta-filter.
"""
from __future__ import absolute_import
from __future__ import division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dconv.layers import DeformConv
from torchdiffeq import odeint as odeint

from core.utils import accuracy
from core.model.metric.metric_model import MetricModel


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        input_ = inputs
        input_ = input_.contiguous().view(input_.size(0), input_.size(1), -1)

        log_probs = self.logsoftmax(input_)
        targets_ = torch.zeros(input_.size(0), input_.size(1)).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )
        targets_ = targets_.unsqueeze(-1)
        targets_ = targets_.cuda()
        loss = (-targets_ * log_probs).mean(0).sum()
        return loss / input_.size(2)


def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [
        nKnovel,
    ]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(
        len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1
    )
    return labels_train_1hot


def shuffle(images, targets, global_targets):
    """
    A trick for META_FILTER training
    """
    sample_num = images.shape[1]
    for i in range(4):
        indices = torch.randperm(sample_num).to(images.device)
        images = images.index_select(1, indices)
        targets = targets.index_select(1, indices)
        global_targets = global_targets.index_select(1, indices)
    return images, targets, global_targets


# Dynamic Sampling
class DynamicWeights_(nn.Module):
    def __init__(self, channels, dilation=1, kernel=3, groups=1):
        super(DynamicWeights_, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

        # padding = 1 if kernel == 3 else 0
        # offset_groups = 1
        self.off_conv = nn.Conv2d(
            channels * 2, 3 * 3 * 2, 5, padding=2, dilation=dilation, bias=False
        )
        self.kernel_conv = DeformConv(
            channels,
            groups * kernel * kernel,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )

        self.K = kernel * kernel
        self.group = groups

    def forward(self, support, query):
        N, C, H, W = support.size()
        # R = C // self.group
        offset = self.off_conv(torch.cat([query, support], 1))  # 学习可变形卷积的偏移量矩阵
        dynamic_filter = self.kernel_conv(support, offset)  # 进行可变形卷积
        dynamic_filter = F.sigmoid(dynamic_filter)
        return dynamic_filter


class DynamicWeights(nn.Module):
    def __init__(self, channels, dilation=1, kernel=3, groups=1, nFeat=640):
        super(DynamicWeights, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        padding = 1 if kernel == 3 else 0
        # offset_groups = 1
        self.unfold = nn.Unfold(
            kernel_size=(kernel, kernel), padding=padding, dilation=1
        )  # 展平操作，将输入特征图中的局部区域展开为列

        self.K = kernel * kernel  # 卷积核的总大小
        self.group = groups  # 组卷积组数
        self.nFeat = nFeat  # 特征数量

    def forward(self, t=None, x=None):
        query, dynamic_filter = x
        N, C, H, W = query.size()
        N_, C_, H_, W_ = dynamic_filter.size()
        R = C // self.group
        # 将动态滤波器重新调整为形状为（-1，self.K）的张量
        dynamic_filter = dynamic_filter.reshape(-1, self.K)

        xd_unfold = self.unfold(query)

        xd_unfold = xd_unfold.contiguous().view(N, C, self.K, H * W)
        xd_unfold = (
            xd_unfold.permute(0, 1, 3, 2)
            .contiguous()
            .view(N, self.group, R, H * W, self.K)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(N * self.group * H * W, R, self.K)
        )  # 调整大小
        # 批量矩阵乘法，对一批矩阵中的每一对进行矩阵乘法。unsqueeze()函数为在指定维度上插入一维
        out1 = torch.bmm(xd_unfold, dynamic_filter.unsqueeze(2))
        out1 = (
            out1.contiguous()
            .view(N, self.group, H * W, R)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(N, self.group * R, H * W)
            .view(N, self.group * R, H, W)
        )

        out1 = F.relu(out1)
        return (out1, torch.zeros([N_, C_, H_, W_]).cuda())


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc  # 定义ODE函数
        self.integration_time = torch.tensor([0, 1]).float()  # 定义积分区间

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(
            x[0]
        )  # 将其数据类型转换为与输入x的类型相同
        out = odeint(
            self.odefunc, x, self.integration_time, rtol=1e-2, atol=1e-2, method="rk4"
        )  # 求解器，rtol和atol为相对和绝对误差限，rk4为四阶龙格库塔方法
        return out[0][1]  # 返回积分结果的第二个时间点的状态

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Model(nn.Module):
    def __init__(self, num_classes=64, nFeat=640, kernel=3, groups=1):
        super(Model, self).__init__()
        self.nFeat = nFeat
        self.global_clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)  # 图中的f_gc

        self.dw_gen = DynamicWeights_(self.nFeat, 1, kernel, groups)
        self.dw = self.dw = ODEBlock(DynamicWeights(self.nFeat, 1, kernel, groups, self.nFeat))

        # 增加ftrain和ftest的维度，其实是做到了统一维度

    def reshape(self, ftrain, ftest):
        b, n1, c, h, w = ftrain.shape
        n2 = ftest.shape[1]
        ftrain = ftrain.unsqueeze(2).repeat(1, 1, n2, 1, 1, 1)
        ftest = ftest.unsqueeze(1).repeat(1, n1, 1, 1, 1, 1)
        return ftrain, ftest

    def get_score(self, ftrain, ftest, num_train, num_test, batch_size):
        b, n2, n1, c, h, w = ftrain.shape

        ftrain_ = ftrain.clone()
        ftest_ = ftest.clone()
        ftrain_ = ftrain_.contiguous().view(-1, *ftrain.size()[3:])
        # 将ftrain_和ftest_变为4维(b*n1*n2,c,h,w)
        ftest_ = ftest_.contiguous().view(-1, *ftest.size()[3:])

        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)  # ftrain归一化处理
        # 将ftrain变为4维(b*n1*n2,c,h,w)
        ftrain_norm = ftrain_norm.reshape(-1, *ftrain_norm.size()[3:])
        # 使用全局平均池化学得一个元分类器
        # 求两次平均，并保持原始维度，得到的是meta_classifier
        conv_weight = ftrain_norm.mean(-1, keepdim=True).mean(-2, keepdim=True)
        # 第一次对每个矩阵的列求平均，第二次对每个矩阵的行求平均，最终化为(b*n1*n2,c,1,1)

        # 用ftrain和ftest进行动态采样学得一个dynamic_filter
        filter_weight = self.dw_gen(ftrain_, ftest_)
        cls_scores = self.dw(x=(ftest_, filter_weight))  # 对动态卷积过程进行神经ODE
        cls_scores = cls_scores.contiguous().view(b * n2, n1, *cls_scores.size()[1:])
        cls_scores = cls_scores.contiguous().view(1, -1, *cls_scores.size()[3:])
        cls_scores = F.conv2d(
            cls_scores, conv_weight, groups=b * n1 * n2, padding=1
        )  # 将计算出的得分与卷积权重卷积
        cls_scores = cls_scores.contiguous().view(b * n2, n1, *cls_scores.size()[2:])
        return cls_scores

    def get_global_pred(self, ftest, ytest, num_test, batch_size, K):
        h = ftest.shape[-1]  # h是ftest的最后一个维度
        # 改变ftest的维度为(batch_size,num_test,K,-1),-1为自适应维度
        ftest_ = ftest.contiguous().view(batch_size, num_test, K, -1)
        ftest_ = ftest_.transpose(2, 3)  # 对ftest_中的每个矩阵进行转置
        ytest_ = ytest.unsqueeze(3)  # 在ytest中加上一维，
        ftest_ = torch.matmul(ftest_, ytest_)  # ftest_和ytest_进行广义张量乘法
        # 改变维度为(batch_size * num_test,-1,h,h)，即每个矩阵变成方阵
        ftest_ = ftest_.contiguous().view(batch_size * num_test, -1, h, h)
        global_pred = self.global_clasifier(ftest_)  # 进行图中的f_gc操作
        return global_pred

    def get_test_score(self, score_list):
        return score_list.mean(-1).mean(-1)

    def forward(self, support_feat, query_feat, support_targets, query_targets, global_labels=None):
        original_feat_shape = support_feat.size()
        batch_size = support_feat.size(0)
        n_support = support_feat.size(1)
        n_query = query_feat.size(1)
        # way_num = support_targets.size(-1)
        K = support_targets.size(2)

        labels_train_transposed = support_targets.transpose(1, 2)

        prototypes = support_feat.contiguous().view(batch_size, n_support, -1)
        prototypes = torch.bmm(labels_train_transposed, prototypes)
        prototypes = prototypes.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
        )
        prototypes = prototypes.contiguous().view(batch_size, -1, *original_feat_shape[2:])
        query_feat = query_feat.contiguous().view(batch_size, n_query, *original_feat_shape[2:])
        prototypes, query_feat = self.reshape(prototypes, query_feat)
        prototypes = prototypes.transpose(1, 2)
        query_feat = query_feat.transpose(1, 2)

        cls_scores = self.get_score(prototypes, query_feat, n_support, n_query, batch_size)

        if not self.training:
            return self.get_test_score(cls_scores)

        global_pred = self.get_global_pred(query_feat, query_targets, n_query, batch_size, K)
        return global_pred, cls_scores


class META_FILTER(MetricModel):
    def __init__(self, num_classes=64, nFeat=640, kernel=3, groups=1, **kwargs):
        super(META_FILTER, self).__init__(**kwargs)
        self.model = Model(num_classes, nFeat, kernel, groups)
        self.criterion = CrossEntropyLoss()

    def set_forward(self, batch):
        images, global_targets = batch
        images = images.to(self.device)
        global_targets = global_targets.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        emb = self.emb_func(images)
        (
            support_feat,
            query_feat,
            support_targets,
            query_targets,
        ) = self.split_by_episode(emb, mode=2)

        # convert to one-hot
        labels_train_1hot = one_hot(support_targets).to(self.device)
        labels_test_1hot = one_hot(query_targets).to(self.device)

        cls_scores = self.model(support_feat, query_feat, labels_train_1hot, labels_test_1hot)

        cls_scores = cls_scores.reshape(episode_size * self.way_num * self.query_num, -1)
        acc = accuracy(cls_scores, query_targets.reshape(-1), topk=1)
        return cls_scores, acc

    def set_forward_loss(self, batch):
        images, global_targets = batch
        images = images.to(self.device)
        global_targets = global_targets.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        emb = self.emb_func(images)
        (
            support_feat,
            query_feat,
            support_targets,
            query_targets,
        ) = self.split_by_episode(emb, mode=2)

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
        labels_train_1hot = one_hot(support_targets).to(self.device)
        labels_test_1hot = one_hot(query_targets).to(self.device)

        ytest, cls_scores = self.model(
            support_feat, query_feat, labels_train_1hot, labels_test_1hot
        )
        # print(ytest.size())
        # print(query_global_targets.size())

        loss1 = self.criterion(ytest, query_global_targets.contiguous().reshape(-1))
        loss2 = self.criterion(cls_scores, query_targets.view(-1))
        loss = loss1 + 0.5 * loss2

        cls_scores = torch.sum(cls_scores.reshape(*cls_scores.size()[:2], -1), dim=-1)
        acc = accuracy(cls_scores, query_targets.reshape(-1), topk=1)
        return cls_scores, acc, loss
