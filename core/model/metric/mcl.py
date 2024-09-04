"""
@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Yang and Zhang, Weifeng and Xiang, Chao and Zheng, Tu and Cai, Deng and He, Xiaofei},
    title     = {Learning To Affiliate: Mutual Centralized Learning for Few-Shot Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14411-14420}
}
Adapted from https://github.com/LouieYang/MCL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metric_model import MetricModel
from core.utils import accuracy


def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))

def l2distance(x, y):
    assert x.shape[:-2] == y.shape[:-2]
    prefix_shape = x.shape[:-2]

    c, M_x = x.shape[-2:]
    M_y = y.shape[-1]
    
    x = x.view(-1, c, M_x)
    y = y.view(-1, c, M_y)

    x_t = x.transpose(1, 2)
    x_t2 = x_t.pow(2.0).sum(-1, keepdim=True)
    y2 = y.pow(2.0).sum(1, keepdim=True)

    ret = x_t2 + y2 - 2.0 * x_t@y
    ret = ret.view(prefix_shape + (M_x, M_y))
    return ret


class Similarity(nn.Module):
    def __init__(self, metric='cosine'):
        super().__init__()
        self.metric = metric

    def forward(self, support_xf, query_xf):
        if query_xf.dim() == 5:
            b, q, c, h, w = query_xf.shape
            query_xf = query_xf.view(b, q, c, h*w)
        else:
            b, q = query_xf.shape[:2]

        s = support_xf.shape[1]

        support_xf = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, s, -1, -1)
        M_q = query_xf.shape[-1]
        M_s = support_xf.shape[-1]

        if self.metric == 'cosine':
            support_xf = _l2norm(support_xf, dim=-2)
            query_xf = _l2norm(query_xf, dim=-2)
            query_xf = torch.transpose(query_xf, 3, 4)
            return query_xf@support_xf # bxQxNxM_qxM_s
        elif self.metric == 'innerproduct':
            query_xf = torch.transpose(query_xf, 3, 4)
            return query_xf@support_xf # bxQxNxM_qxM_s
        elif self.metric == 'euclidean':
            return 1 - l2distance(query_xf, support_xf)
        elif self.metric == 'neg_ed':
            query_xf = query_xf.contiguous().view(-1, c, M_q).transpose(-2, -1).contiguous()
            support_xf = support_xf.contiguous().view(-1, c, M_s).transpose(-2, -1).contiguous()
            dist = torch.cdist(query_xf, support_xf)
            return -dist.view(b, q, s, M_q, M_s) / 2.
        else:
            raise NotImplementedError

class MCLMask(nn.Module):
    def __init__(self, katz_factor, gamma, gamma2):

        super().__init__()
        self.inner_simi = Similarity(metric='cosine')
        self.gamma = gamma
        self.gamma2 = gamma2
        self.katz_factor = katz_factor

    def forward(self, support_xf, query_xf, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot

        b, s, c, h, w = support_xf.shape
        q = query_xf.shape[1]
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
        support_xf = support_xf.view(b, self.n_way, c, h * w)
        S = self.inner_simi(support_xf, query_xf)
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        N_examples = b * q
        St = S.transpose(-2, -1)
        device = S.device

        T_sq = torch.exp(self.gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True)
        T_qs = torch.exp(self.gamma2 * (St - St.max(-1, keepdim=True)[0]))
        T_qs = T_qs / T_qs.sum(-1, keepdim=True)

        T = torch.cat([
            torch.cat([torch.zeros((N_examples, M_s, M_s), device=device), T_sq.transpose(-2, -1)], dim=-1),
            torch.cat([T_qs.transpose(-2, -1), torch.zeros((N_examples, M_q, M_q), device=device)], dim=-1),
        ], dim=-2)

        katz = (torch.inverse(torch.eye(M_s + M_q, device=device)[None].repeat(N_examples, 1, 1) - self.katz_factor * T) - \
                torch.eye(M_s + M_q, device=S.device)[None].repeat(N_examples, 1, 1))@torch.ones((N_examples, M_s + M_q, 1), device=device)
        katz_query = katz.squeeze(-1)[:, M_s:] / katz.squeeze(-1)[:, M_s:].sum(-1, keepdim=True)
        # katz_support = katz.squeeze(-1)[:, :M_s] / katz.squeeze(-1)[:, :M_s].sum(-1, keepdim=True)
        # katz_support = katz_support.view(b, q, self.n_way, -1)
        # katz_support = katz_support / katz_support.sum(-1, keepdim=True)
        # katz_support = katz_support.view(b, q, self.n_way, h, w).unsqueeze(3)
        katz_query = katz_query.view(b, q, h, w).unsqueeze(2)
        return katz_query

class MCLLayer(nn.Module):
    def __init__(self, n_k, katz_factor, gamma, gamma2):
        super(MCLLayer, self).__init__()
        self.n_k = n_k
        self.gamma = gamma
        self.gamma2 = gamma2
        self.katz_factor = katz_factor
        self.inner_simi = Similarity(metric='cosine')
        self.criterion = nn.NLLLoss()
        

    def averaging_based_similarities(self, support_xf, support_y, query_xf, query_y):
        b, s, c, h, w = support_xf.shape
        q = query_xf.shape[1]
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).mean(2)
        support_xf = support_xf.view(b, self.n_way, c, h * w)
        S = self.inner_simi(support_xf, query_xf) 
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        return S

    def bipartite_katz_forward(self, support_xf, support_y, query_xf, query_y, similarity_f):
        katz_factor = self.katz_factor
        S = similarity_f(support_xf, support_y, query_xf, query_y)
        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)
        device = S.device

        T_sq = torch.exp(self.gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True)
        T_qs = torch.exp(self.gamma2 * (St - St.max(-1, keepdim=True)[0])) 
        T_qs = T_qs / T_qs.sum(-1, keepdim=True)

        T = torch.cat([
            torch.cat([torch.zeros((N_examples, M_s, M_s), device=device), T_sq.transpose(-2, -1)], dim=-1),
            torch.cat([T_qs.transpose(-2, -1), torch.zeros((N_examples, M_q, M_q), device=device)], dim=-1),
        ], dim=-2)
        katz = (torch.inverse(torch.eye(M_s + M_q, device=device)[None].repeat(N_examples, 1, 1) - katz_factor * T) - \
                torch.eye(M_s + M_q, device=S.device)[None].repeat(N_examples, 1, 1))@torch.ones((N_examples, M_s + M_q, 1), device=device)
        partial_katz = katz.squeeze(-1)[:, :M_s] / katz.squeeze(-1)[:, :M_s].sum(-1, keepdim=True)
        predicts = partial_katz.view(N_examples, self.n_way, -1).sum(-1)
        return predicts

    def forward(self, support_xf, support_y, query_xf, query_y, n_way, k_shot):
        self.n_way = n_way
        self.k_shot = k_shot
        return self.bipartite_katz_forward(support_xf, support_y, query_xf, query_y, self.averaging_based_similarities)

class MCL(MetricModel):
    def __init__(self, n_k=3, **kwargs):
        super(MCL, self).__init__(**kwargs)
        self.mcl_layer = MCLLayer(n_k,kwargs.get('katz_factor'),kwargs.get('gamma'),kwargs.get('gamma2'))
        self.loss_func = nn.NLLLoss()

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )
        output = self.mcl_layer(
            support_feat, support_target, query_feat, query_target, self.way_num, self.shot_num,
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )
        output = self.mcl_layer(
            support_feat, support_target, query_feat, query_target, self.way_num, self.shot_num,
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))
        loss = self.loss_func(torch.log(output), query_target.reshape(-1))
        return output, acc, loss