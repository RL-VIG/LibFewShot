import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from core.utils import accuracy
from .metric_model import MetricModel


class FRNLayer(nn.Module):
    def __init__(
        self,
        num_cat=None,
        num_channel=640,
    ):
        super().__init__()
        self.resolution = 25
        self.d = num_channel
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)

    def forward(self, support, query, way_num, shot_num, query_num):
        n2g_l2_dist = self.get_neg_l2_dist(
            support, query, way_num, shot_num, query_num
        )
        logits = n2g_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=2)
        return log_prediction

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        # query: n, way*query_shot*resolution, d
        # support: n, way, shot*resolution, d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(2) / support.size(3)

        # correspond to lambda in the paper
        lam = reg * alpha.exp() + 1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0, 1, 3, 2)  # n, way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper

            sts = st.matmul(support)  # n, way, d, d
            m_inv = (
                sts
                + torch.eye(sts.size(-1))
                .to(sts.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .mul(lam)
            ).inverse()  # n, way, d, d
            hat = m_inv.matmul(sts)  # n, way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st)  # n, way, shot*resolution, shot*resolution
            m_inv = (
                sst
                + torch.eye(sst.size(-1))
                .to(sst.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .mul(lam)
            ).inverse()  # n, way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support)  # n, way, d, d
        Q_bar = query.unsqueeze(1).matmul(hat).mul(rho)  # n, way, way*query_shot*resolution, d
        dist = (
            (Q_bar - query.unsqueeze(1)).pow(2).sum(3).permute(0, 2, 1)
        )  # n, way*query_shot*resolution, way
        return dist

    def get_neg_l2_dist(self, support, query, way, shot, query_shot):
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = self.get_recon_dist(
            query=query, support=support, alpha=alpha, beta=beta
        )  # way*query_shot*resolution, way
        neg_l2_dist = (
            recon_dist.neg().view(-1, way * query_shot, resolution, way).mean(2)
        )  # way*query_shot, way

        return neg_l2_dist


class FRN(MetricModel):
    def __init__(self, **kwargs):
        super(FRN, self).__init__(**kwargs)
        self.frn_layer = FRNLayer()
        self.loss_func = nn.NLLLoss().cuda()

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
        episode_size, _, c, h, w = support_feat.size()
        support_feat = (
            support_feat.view(episode_size, self.way_num, self.shot_num, c, h * w)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
            .view(episode_size, self.way_num, -1, c)
        )
        query_feat = (
            query_feat.permute(0, 1, 3, 4, 2).contiguous().view(episode_size, -1, c)
        )
        output = self.frn_layer(
            support_feat, query_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(-1, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        feat = feat/np.sqrt(640)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )
        episode_size, _, c, h, w = support_feat.size()
        support_feat = (
            support_feat.view(episode_size, self.way_num, self.shot_num, c, h * w)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
            .view(episode_size, self.way_num, -1, c)
        )
        query_feat = (
            query_feat.permute(0, 1, 3, 4, 2).contiguous().view(episode_size, -1, c)
        )
        output = self.frn_layer(
            support_feat, query_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(-1, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))
        loss1 = self.loss_func(output, query_target.reshape(-1))
        loss2 = auxrank(support_feat)
        loss = loss1 + loss2.mean()
        return output, acc, loss


def auxrank(support):
    way = support.size(1)
    shot = support.size(2)
    support = support / support.norm(2).unsqueeze(-1)
    L1 = torch.zeros((way**2 - way) // 2).long().cuda()
    L2 = torch.zeros((way**2 - way) // 2).long().cuda()
    counter = 0
    for i in range(way):
        for j in range(i):
            L1[counter] = i
            L2[counter] = j
            counter += 1
    s1 = support.index_select(1, L1)  # (s^2-s)/2, s, d
    s2 = support.index_select(1, L2)  # (s^2-s)/2, s, d
    dists = s1.matmul(s2.permute(0, 1, 3, 2))  # (s^2-s)/2, s, s
    assert dists.size(-1) == shot
    frobs = dists.pow(2).sum(-1).sum(-1)
    return frobs.sum(-1).mul(0.03)
