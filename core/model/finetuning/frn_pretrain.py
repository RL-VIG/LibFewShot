import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from core.utils import accuracy
from .finetuning_model import FinetuningModel


class FRNLayer(nn.Module):
    def __init__(
        self,
        num_cat=64,
        num_channel=640,
    ):
        super().__init__()
        self.resolution = 25
        self.d = num_channel
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.num_cat = num_cat
        # category matrix, correspond to matrix M of section 3.6 in the paper
        self.cat_mat = nn.Parameter(
            torch.randn(self.num_cat, self.resolution, self.d), requires_grad=True
        )

    def forward_test(self, support, query, way_num, shot_num, query_num):
        n2g_l2_dist = self.get_neg_l2_dist(support, query, way_num, shot_num, query_num)
        logits = n2g_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction

    def forward(self, feat, batch_size):
        alpha = self.r[0]
        beta = self.r[1]
        recon_dist = self.get_recon_dist(
            query=feat, support=self.cat_mat, alpha=alpha, beta=beta
        )
        neg_l2_dist = (
            recon_dist.neg().view(batch_size, self.resolution, self.num_cat).mean(1)
        )  # batch_size,num_cat

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction
        pass

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        # query: n, way*query_shot*resolution, d
        # support: n, way, shot*resolution, d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1) / support.size(2)

        # correspond to lambda in the paper
        lam = reg * alpha.exp() + 1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0, 2, 1)  # n, way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper

            sts = st.matmul(support)  # n, way, d, d
            m_inv = (
                sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)
            ).inverse()  # n, way, d, d
            hat = m_inv.matmul(sts)  # n, way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st)  # n, way, shot*resolution, shot*resolution
            m_inv = (
                sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)
            ).inverse()  # n, way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support)  # n, way, d, d

        Q_bar = query.matmul(hat).mul(rho)  # n, way, way*query_shot*resolution, d

        dist = (
            (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)
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
            recon_dist.neg().view(way * query_shot, resolution, way).mean(1)
        )  # way*query_shot, way

        return neg_l2_dist


class FRN_Pretrain(FinetuningModel):
    def __init__(self, **kwargs):
        print(kwargs)
        super(FRN_Pretrain, self).__init__(**kwargs)
        self.frn_layer = FRNLayer(self.num_cat, self.num_channel)
        self.loss_func = nn.NLLLoss().cuda()

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        feat = feat / np.sqrt(640)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=3
        )
        _, c, h, w = support_feat.size()
        support_feat = (
            support_feat.view(self.way_num * self.shot_num, c, h * w)
            .permute(0, 2, 1)
            .contiguous()
            .view(self.way_num, -1, c)
        )
        query_feat = (
            query_feat.view(self.way_num * self.query_num, c, h * w)
            .permute(0, 2, 1)
            .contiguous()
            .view(-1, c)
        )
        output = self.frn_layer.forward_test(
            support_feat, query_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(-1, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image, global_target = image.to(self.device), global_target.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        feat = feat / np.sqrt(640)
        batch_size = feat.size(0)
        feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, 640)
        output = self.frn_layer(feat, batch_size).reshape(batch_size, -1)
        acc = accuracy(output, global_target.reshape(-1))
        loss = self.loss_func(output, global_target.reshape(-1))
        return output, acc, loss
    
    def meta_set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        feat = feat / np.sqrt(640)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=3
        )
        _, c, h, w = support_feat.size()
        support_feat = (
            support_feat.view(self.way_num * self.shot_num, c, h * w)
            .permute(0, 2, 1)
            .contiguous()
            .view(self.way_num, -1, c)
        )
        query_feat = (
            query_feat.view(self.way_num * self.query_num, c, h * w)
            .permute(0, 2, 1)
            .contiguous()
            .view(-1, c)
        )
        output = self.frn_layer(
            support_feat, query_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(-1, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc
