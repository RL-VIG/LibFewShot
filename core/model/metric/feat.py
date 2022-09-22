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
import torch.nn.functional as F
from torch import nn
import numpy as np
from core.utils import accuracy
from .metric_model import MetricModel


class ProtoLayer(nn.Module):
    """
    This Proto Layer is different from Proto_layer @ ProtoNet
    """

    def __init__(self, way_num, shot_num, query_num):
        super(ProtoLayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num

    def forward(self, query, proto, mode="euclidean", temperature=1.0):
        return {
            # t, wq, 1, c - t, 1, w, c -> t, wq, w
            "euclidean": lambda x, y: -torch.sum(
                torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
                dim=3,
            )
            / temperature,
            # t, wq, c - t, c, w -> t, wq, w
            "cos_sim": lambda x, y: torch.mm(
                F.normalize(x, p=2, dim=-1),
                torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # FEAT did not normalize the query_feat
            )
            / temperature,
        }[mode](query, proto)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).reshape(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).reshape(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).reshape(sz_b, len_v, n_head, d_v)

        q = (
            q.permute(2, 0, 1, 3).contiguous().reshape(-1, len_q, d_k)
        )  # (n*b) x lq x dk
        k = (
            k.permute(2, 0, 1, 3).contiguous().reshape(-1, len_k, d_k)
        )  # (n*b) x lk x dk
        v = (
            v.permute(2, 0, 1, 3).contiguous().reshape(-1, len_v, d_v)
        )  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.reshape(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().reshape(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FEAT(MetricModel):
    def __init__(self, hdim, temperature, temperature2, balance, mode, **kwargs):
        super(FEAT, self).__init__(**kwargs)
        self.mode = mode
        self.balance = balance
        self.hdim = hdim
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.loss_func = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.temperature2 = temperature2
        self.proto_layer = ProtoLayer(self.way_num, self.shot_num, self.query_num)

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        self.episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        self.feat = self.emb_func(images)  # [e*(q+s) x hdim]
        (
            self.support_feat,
            self.query_feat,
            support_target,
            query_target,
        ) = self.split_by_episode(self.feat, mode=1)

        logits = self._calc_logits().reshape(-1, self.way_num)

        acc = accuracy(logits, query_target.reshape(-1))
        return logits, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        self.episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        self.feat = self.emb_func(images)  # [e*(q+s) x hdim]
        (
            self.support_feat,
            self.query_feat,
            support_target,
            query_target,
        ) = self.split_by_episode(self.feat, mode=1)

        target_aux = torch.cat(
            [
                support_target.reshape(-1).contiguous(),
                query_target.reshape(-1).contiguous(),
            ]
        )

        logits = self._calc_logits().reshape(-1, self.way_num)
        reg_logits = self._calc_reg_logits().reshape(-1, self.way_num)

        loss1 = self.loss_func(logits, query_target.reshape(-1))

        loss_reg = self.loss_func(reg_logits, target_aux)

        acc = accuracy(logits, query_target.reshape(-1))
        loss = loss1 * self.balance + loss_reg
        return logits, acc, loss

    def _calc_logits(self):
        """
        support -> proto
        query and proto
        """
        support_feat = self.support_feat.reshape(
            self.episode_size, self.way_num, self.shot_num, -1
        ).mean(dim=2)
        proto = self.slf_attn(support_feat, support_feat, support_feat)
        # proto e w dim
        # query e wq hdim
        # num_batch = e
        # num_proto = way
        # num_query = way * hdim ?
        logits = self.proto_layer(self.query_feat, proto, self.mode, self.temperature)

        return logits

    def _calc_reg_logits(self):
        """
        aux_task -> query
        aux_center = proto
        """
        aux_task = self.feat.reshape(
            self.episode_size, self.way_num, self.shot_num + self.query_num, -1
        )
        # e w sq d
        num_query = np.prod(aux_task.shape[1:3])  # wsq*d
        aux_task = aux_task.reshape(
            self.episode_size * self.way_num, self.shot_num + self.query_num, self.hdim
        )

        # apply the transformation over the Aug Task
        aux_emb = self.slf_attn(aux_task, aux_task, aux_task).reshape(
            self.episode_size, self.way_num, self.shot_num + self.query_num, self.hdim
        )  # e w qs d
        # compute class mean
        aux_center = aux_emb.mean(2)  # e w d # same as proto
        aux_task = (
            aux_task.reshape(
                self.episode_size * self.way_num * (self.shot_num + self.query_num),
                self.hdim,
            )
            .contiguous()
            .unsqueeze(1)
        )
        aux_center = aux_center.expand(
            self.episode_size * num_query, self.way_num, self.hdim
        )

        # proto ewsq w d
        # query ewsq 1 d
        logits_reg = self.proto_layer(
            aux_task, aux_center, self.mode, self.temperature2
        )

        return logits_reg
