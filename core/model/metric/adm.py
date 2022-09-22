# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/ijcai/LiWHSGL20,
  author    = {Wenbin Li and
               Lei Wang and
               Jing Huo and
               Yinghuan Shi and
               Yang Gao and
               Jiebo Luo},
  title     = {Asymmetric Distribution Measure for Few-shot Learning},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2020},
  pages     = {2957--2963},
  year      = {2020},
  url       = {https://doi.org/10.24963/ijcai.2020/409},
  doi       = {10.24963/ijcai.2020/409}
}
https://arxiv.org/abs/2002.00153

Adapted from https://github.com/WenbinLee/ADM.
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class ADMLayer(nn.Module):
    def __init__(self, way_num, shot_num, query_num, n_k, device):
        super(ADMLayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.n_k = n_k
        self.device = device
        self.normLayer = nn.BatchNorm1d(self.way_num * 2, affine=True)
        self.fcLayer = nn.Conv1d(1, 1, kernel_size=2, stride=1, dilation=5, bias=False)

    def _cal_cov_matrix_batch(self, feat):  # feature: e *  Batch * descriptor_num * 64
        e, _, n_local, c = feat.size()
        feature_mean = torch.mean(feat, 2, True)  # e * Batch * 1 * 64
        feat = feat - feature_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)  # ebc1 * eb1c = ebcc
        cov_matrix = torch.div(cov_matrix, n_local - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(
            self.device
        )  # broadcast from the last dim

        return feature_mean, cov_matrix

    def _cal_cov_batch(self, feat):  # feature: e * 25 * 64 * 21 * 21
        e, b, c, h, w = feat.size()
        feat = feat.reshape(e, b, c, -1).permute(0, 1, 3, 2)
        feat_mean = torch.mean(feat, 2, True)  # e * Batch * 1 * 64
        feat = feat - feat_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
        cov_matrix = torch.div(cov_matrix, h * w - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(self.device)

        return feat_mean, cov_matrix

    def _calc_kl_dist_batch(self, mean1, cov1, mean2, cov2):
        """

        :param mean1: e * 75 * 1 * 64
        :param cov1: e * 75 * 64 * 64
        :param mean2: e * 5 * 1 * 64
        :param cov2: e * 5 * 64 * 64
        :return:
        """

        cov2_inverse = torch.inverse(cov2)  # e * 5 * 64 * 64
        mean_diff = -(mean1 - mean2.squeeze(2).unsqueeze(1))  # e * 75 * 5 * 64

        # Calculate the trace
        matrix_prod = torch.matmul(
            cov1.unsqueeze(2), cov2_inverse.unsqueeze(1)
        )  # e * 75 * 5 * 64 * 64
        # trace_dist = [[torch.trace(matrix_prod[e][j][i]).unsqueeze(0) # modified for multi-task, stack for 64*64
        # tensors
        #                for j in range(matrix_prod.size(1))
        #                for i in range(matrix_prod.size(2))]
        #               for e in range(matrix_prod.size(0))] # list of trace_dist
        # trace_dist = torch.stack([torch.cat(trace_dist_list, 0) for trace_dist_list in trace_dist]) #
        trace_dist = torch.diagonal(
            matrix_prod, offset=0, dim1=-2, dim2=-1
        )  # e * 75 * 5 * 64
        trace_dist = torch.sum(trace_dist, dim=-1)  # e * 75 * 5
        # trace_dist = trace_dist.view(matrix_prod.size(0),matrix_prod.size(1), matrix_prod.size(2))  # e * 75 * 5

        # Calcualte the Mahalanobis Distance
        maha_prod = torch.matmul(
            mean_diff.unsqueeze(3), cov2_inverse.unsqueeze(1)
        )  # e * 75 * 5 * 1 * 64
        maha_prod = torch.matmul(
            maha_prod, mean_diff.unsqueeze(4)
        )  # e * 75 * 5 * 1 * 1
        maha_prod = maha_prod.squeeze(4)
        maha_prod = maha_prod.squeeze(3)  # e * 75 * 5

        matrix_det = torch.slogdet(cov2).logabsdet.unsqueeze(1) - torch.slogdet(
            cov1
        ).logabsdet.unsqueeze(2)

        kl_dist = trace_dist + maha_prod + matrix_det - mean1.size(3)

        return kl_dist / 2.0

    # Calculate KL divergence Distance
    def _cal_adm_sim(self, query_feat, support_feat):
        """

        :param query_feat: e * 75 * 64 * 21 * 21
        :param support_feat: e * 25 * 64 * 21 * 21
        :return:
        """
        # query_mean: e * 75 * 1 * 64  query_cov: e * 75 * 64 * 64
        e, b, c, h, w = query_feat.size()
        e, s, _, _, _ = support_feat.size()

        query_mean, query_cov = self._cal_cov_batch(query_feat)

        query_feat = query_feat.reshape(e, b, c, -1).permute(0, 1, 3, 2).contiguous()

        # Calculate the mean and covariance of the support set
        support_feat = (
            support_feat.reshape(e, s, c, -1).permute(0, 1, 3, 2).contiguous()
        )
        support_set = support_feat.reshape(e, self.way_num, self.shot_num * h * w, c)

        # s_mean: e * 5 * 1 * 64  s_cov: e * 5 * 64 * 64
        s_mean, s_cov = self._cal_cov_matrix_batch(support_set)

        # Calculate the Wasserstein Distance
        kl_dis = -self._calc_kl_dist_batch(
            query_mean, query_cov, s_mean, s_cov
        )  # e * 75 * 5

        # Calculate the Image-to-Class Similarity
        query_norm = F.normalize(query_feat, p=2, dim=3)
        support_norm = F.normalize(support_feat, p=2, dim=3)
        support_norm = support_norm.reshape(e, self.way_num, self.shot_num * h * w, c)

        # cosine similarity between a query set and a support set
        # e * 75 * 5 * 441 * 2205
        inner_prod_matrix = torch.matmul(
            query_norm.unsqueeze(2),
            support_norm.permute(0, 1, 3, 2).unsqueeze(1),
        )

        # choose the top-k nearest neighbors
        # e * 75 * 5 * 441 * 1
        topk_value, topk_index = torch.topk(inner_prod_matrix, self.n_k, 4)
        inner_sim = torch.sum(torch.sum(topk_value, 4), 3)  # e * 75 * 5

        # Using FC layer to combine two parts ---- The original
        adm_sim_soft = torch.cat((kl_dis, inner_sim), 2)

        adm_sim_soft = torch.cat(
            [self.normLayer(each_task).unsqueeze(1) for each_task in adm_sim_soft]
        )
        # e * 75 * 1 * 10

        adm_sim_soft = self.fcLayer(adm_sim_soft).squeeze(1).reshape([e, b, -1])

        return adm_sim_soft

    def forward(self, query_feat, support_feat):
        return self._cal_adm_sim(query_feat, support_feat)


class ADM(MetricModel):
    def __init__(self, n_k=3, **kwargs):
        super(ADM, self).__init__(**kwargs)
        self.n_k = n_k
        self.adm_layer = ADMLayer(
            self.way_num, self.shot_num, self.query_num, n_k, self.device
        )
        self.loss_func = nn.CrossEntropyLoss()

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
            feat, mode=2
        )

        output = self.adm_layer(query_feat, support_feat).reshape(
            episode_size * self.way_num * self.query_num, -1
        )
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
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
            feat, mode=2
        )
        # assume here we will get n_dim=5
        output = self.adm_layer(query_feat, support_feat).reshape(
            episode_size * self.way_num * self.query_num, -1
        )
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc, loss
