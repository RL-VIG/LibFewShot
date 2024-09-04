# -*- coding: utf-8 -*-
'''
@inproceedings{NEURIPS2023_9b013332,
 author = {Zheng, Kaipeng and Zhang, Huishuai and Huang, Weiran},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {49403--49415},
 publisher = {Curran Associates, Inc.},
 title = {DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall\textquotesingle s Rank Correlation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/9b01333262789ea3a65a5fab4c22feae-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
https://arxiv.org/abs/2307.15317

Adapter from https://github.com/kaipengm2/DiffKendall
'''
import torch
from torch import nn
import torch.nn.functional as F
from core.utils import accuracy
from .finetuning_model import FinetuningModel
from itertools import combinations
import random

def compute_c_pair(c):
    return list(combinations(list(range(c)), 2))

def diffkendall(support, query, c_pair, beta=1, T=0.0125):
    '''The differentiable Kendall's rank correlation.'''
    support_prank = support[:, c_pair].diff().squeeze()
    query_prank = query[:, c_pair].diff().squeeze(-1)
    score = support_prank.repeat([query.shape[0], 1, 1]) * query_prank.unsqueeze(1).repeat([1, support.shape[0], 1])
    score = 1 / (1 + (-score * beta).exp())
    score = 2 * score - 1
    score = score.mean(dim=-1)
    score = score / T
    return score

def diffkendall_for_batches(support, query, beta=1, T=0.0125):
    c = support.shape[-1]
    c_pair = compute_c_pair(c)
    batches = support.shape[0]
    scores = [diffkendall(support[i], query[i], c_pair, beta, T) for i in range(batches)]
    return torch.stack(scores)

def kendall_ranking_correlation(support, query, c_pair):
    '''Kendall's rank correlation.'''
    support_prank = support[:, c_pair].diff(dim=-1).sign().squeeze()
    query_prank = query[:, c_pair].diff(dim=-1).sign().squeeze()
    score = torch.mm(query_prank, support_prank.T)
    return score / len(c_pair)

def kendall_ranking_correlation_for_batches(support, query):
    c = support.shape[-1]
    c_pair = compute_c_pair(c)
    batches = support.shape[0]
    
    scores = [kendall_ranking_correlation(support[i], query[i], c_pair) for i in range(batches)]
    return torch.stack(scores)



# def compute_c_pair(c):
#     return list(combinations(list(range(c)), 2))

# def kendall_ranking_correlation(support, query, c_pair):
#     '''Kendall's rank correlation.'''
#     support_prank = support[:, c_pair].diff(dim=-1).sign().squeeze()
#     query_prank = query[:, c_pair].diff(dim=-1).sign().squeeze()
#     score = torch.mm(query_prank, support_prank.T)
#     return score / len(c_pair)

# def kendall_ranking_correlation_for_batches(support, query):
#     c = support.shape[-1]
#     c_pair = compute_c_pair(c)
#     c_pair_tensor = torch.tensor(c_pair, dtype=torch.long)
#     support_prank = support[:, :, c_pair_tensor].diff(dim=-1).sign()
#     query_prank = query[:, :, c_pair_tensor].diff(dim=-1).sign()
#     support_prank = support_prank.view(support.shape[0], -1, len(c_pair))
#     query_prank = query_prank.view(query.shape[0], -1, len(c_pair))
#     scores = torch.bmm(query_prank, support_prank.transpose(1, 2)) / len(c_pair)
#     return scores

class ProtoLayer(nn.Module):
    def __init__(self):
        super(ProtoLayer, self).__init__()

    def forward(
        self,
        query_feat,
        proto_feat,
        mode="kendall",
    ):

        return {
            # t, wq, 1, c - t, 1, w, c -> t, wq, w
            "euclidean": lambda x, y: -torch.sum(
                torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
                dim=3,
            ),
            # t, wq, c - t, c, w -> t, wq, w
            "cos_sim": lambda x, y: torch.matmul(
                F.normalize(x, p=2, dim=-1),
                torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # FEAT did not normalize the query_feat
            ),
            "kendall":lambda x, y: kendall_ranking_correlation_for_batches(y, x)
        }[mode](query_feat, proto_feat)
        
class MetabaselineKendallPretrain(FinetuningModel):
    def __init__(self, feat_dim, num_class, **kwargs):
        super(MetabaselineKendallPretrain, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class

        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.proto_layer = ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
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
        episode_size = support_feat.size(0)
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()
        query_feat = query_feat.reshape(t, self.way_num * self.query_num, c)
        support_feat = support_feat.reshape(t, self.way_num, self.shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)
        output = self.proto_layer(
            query_feat, proto_feat, "kendall"
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)
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
        output = self.classifier(feat)
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss

    