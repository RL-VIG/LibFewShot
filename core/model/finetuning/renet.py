# -*- coding: utf-8 -*-
"""
@article{DBLP:journals/corr/abs-2108-09666,
  author    = {Dahyun Kang and
               Heeseung Kwon and
               Juhong Min and
               Minsu Cho},
  title     = {Relational Embedding for Few-Shot Classification},
  journal   = {CoRR},
  volume    = {abs/2108.09666},
  year      = {2021},
}
https://arxiv.org/abs/2108.09666

Adapted from https://github.com/dahyun-kang/renet/.
"""
import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from core.utils.enum_type import ModelType

# CCA Module #
""" code references: https://github.com/ignacio-rocco/ncnet and https://github.com/gengshan-y/VCN """


class CCA(nn.Module):
    def __init__(self, kernel_sizes=[3, 3], planes=[16, 1]):
        super(CCA, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()

        for i in range(num_layers):
            ch_in = 1 if i == 0 else planes[i - 1]
            ch_out = planes[i]
            k_size = kernel_sizes[i]
            nn_modules.append(
                SepConv4d(
                    in_planes=ch_in, out_planes=ch_out, ksize=k_size, do_padding=True
                )
            )
            if i != num_layers - 1:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
        # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
        # because of the ReLU layers in between linear layers,
        # this operation is different than convolving a single time with the filters+filters^T
        # and therefore it makes sense to do this.
        x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(
            0, 1, 4, 5, 2, 3
        )
        return x


class SepConv4d(nn.Module):
    """approximates 3 x 3 x 3 x 3 kernels via two subsequent 3 x 3 x 1 x 1 and 1 x 1 x 3 x 3"""

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=(1, 1, 1),
        ksize=3,
        do_padding=True,
        bias=False,
    ):
        super(SepConv4d, self).__init__()
        self.isproj = False
        padding1 = (0, ksize // 2, ksize // 2) if do_padding else (0, 0, 0)
        padding2 = (ksize // 2, ksize // 2, 0) if do_padding else (0, 0, 0)

        if in_planes != out_planes:
            self.isproj = True
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=1,
                    bias=bias,
                    padding=0,
                ),
                nn.BatchNorm2d(out_planes),
            )

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_planes,
                out_channels=in_planes,
                kernel_size=(1, ksize, ksize),
                stride=stride,
                bias=bias,
                padding=padding1,
            ),
            nn.BatchNorm3d(in_planes),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_planes,
                out_channels=in_planes,
                kernel_size=(ksize, ksize, 1),
                stride=stride,
                bias=bias,
                padding=padding2,
            ),
            nn.BatchNorm3d(in_planes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape

        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x


# SCR Module #


class SCR(nn.Module):
    def __init__(
        self,
        planes=[640, 64, 64, 64, 640],
        stride=(1, 1, 1),
        ksize=3,
        do_padding=False,
        bias=False,
    ):
        super(SCR, self).__init__()
        self.ksize = (ksize,) * 4 if isinstance(ksize, int) else ksize
        padding1 = (
            (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)
        )

        self.conv1x1_in = nn.Sequential(
            nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[1]),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                planes[1],
                planes[2],
                (1, self.ksize[2], self.ksize[3]),
                stride=stride,
                bias=bias,
                padding=padding1,
            ),
            nn.BatchNorm3d(planes[2]),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                planes[2],
                planes[3],
                (1, self.ksize[2], self.ksize[3]),
                stride=stride,
                bias=bias,
                padding=padding1,
            ),
            nn.BatchNorm3d(planes[3]),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]),
        )

    def forward(self, x):
        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)

        x = self.conv1x1_in(x)  # [80, 640, hw, 25] -> [80, 64, HW, 25]

        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
        x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]

        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]
        return x


class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x

        x = self.unfold(x)  # b, cuv, h, w
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        x = x * identity.unsqueeze(2).unsqueeze(
            2
        )  # b, c, u, v, h, w * b, c, 1, 1, h, w
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v
        return x


class SCRLayer(nn.Module):
    def __init__(self, planes=[640, 64, 64, 64, 640]):
        super(SCRLayer, self).__init__()
        kernel_size = (5, 5)
        padding = 2
        stride = (1, 1, 1)
        self.model = nn.Sequential(
            SelfCorrelationComputation(kernel_size=kernel_size, padding=padding),
            SCR(planes=planes, stride=stride),
        )

    def forward(self, x):  # b/ewsq c h w
        return self.model(x)


class CCALayer(nn.Module):
    def __init__(
        self, feat_dim, way_num, shot_num, query_num, temperature, temperature_attn
    ):
        super(CCALayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.temperature = temperature
        self.temperature_attn = temperature_attn

        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(feat_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def get_4d_correlation_map(self, spt, qry):
        """
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: way * C * H_s * W_s
        :param qry: num_qry * C * H_q * W_q
        :return: 4d correlation tensor: num_qry * way * H_s * W_s * H_q * W_q
        :rtype:
        """
        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        # num_way * C * H_p * W_p --> num_qry * way * H_p * W_p
        # num_qry * C * H_q * W_q --> num_qry * way * H_q * W_q
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map = torch.einsum("qncij,qnckl->qnijkl", spt, qry)
        return similarity_map

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def forward(self, spt, qry):

        spt = spt.squeeze(0)

        # shifting channel activations by the channel mean
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        # (S * C * Hs * Ws, Q * C * Hq * Wq) -> Q * S * Hs * Ws * Hq * Wq
        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        # corr4d refinement
        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)

        # applying softmax for each side
        corr4d_s = F.softmax(corr4d_s / self.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)

        # suming up matching scores
        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])

        # applying attention
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)

        # averaging embeddings for k > 1 shots
        if self.shot_num > 1:
            spt_attended = spt_attended.view(
                num_qry, self.way_num, self.shot_num, *spt_attended.shape[2:]
            )
            qry_attended = qry_attended.view(
                num_qry, self.way_num, self.shot_num, *qry_attended.shape[2:]
            )
            spt_attended = spt_attended.mean(dim=2)
            qry_attended = qry_attended.mean(dim=2)

        # In the main paper, we present averaging in Eq.(4) and summation in Eq.(5).
        # In the implementation, the order is reversed, however, those two ways become eventually the same anyway :)
        spt_attended = spt_attended.mean(dim=[-1, -2])
        qry_attended = qry_attended.mean(dim=[-1, -2])
        qry_pooled = qry.mean(dim=[-1, -2])

        similarity_matrix = F.cosine_similarity(spt_attended, qry_attended, dim=-1)

        return similarity_matrix / self.temperature, qry_pooled


class RENet(FinetuningModel):
    def __init__(
        self, feat_dim, lambda_epi, temperature, temperature_attn, num_classes, **kwargs
    ):
        super(RENet, self).__init__(**kwargs)
        self.lambda_epi = lambda_epi
        self.temperature = temperature
        self.temperature_attn = temperature_attn

        self.fc = nn.Linear(feat_dim, num_classes)
        self.scr_layer = SCRLayer(planes=[feat_dim, 64, 64, 64, feat_dim])
        self.cca_layer = CCALayer(
            feat_dim,
            self.way_num,
            self.shot_num,
            self.query_num,
            self.temperature,
            self.temperature_attn,
        )

        self.loss_func = nn.CrossEntropyLoss()

    def encode(self, x):
        x = self.emb_func(x)

        identity = x
        x = self.scr_layer(x)
        x = x + identity
        x = F.relu(x, inplace=False)

        return x

    @torch.no_grad()
    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        ep_images, _ = batch

        ep_images = ep_images.to(self.device)  # ew(qs) c h w

        # extract features
        ep_feat = self.encode(ep_images)

        # CCA for ep_images
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            ep_feat, mode=2
        )
        # ws c h w ; wq c h w
        _, _, c, h, w = support_feat.shape
        support_feat = support_feat.reshape([-1, c, h, w])
        query_feat = query_feat.reshape([-1, c, h, w])
        logits, qry_pooled = self.cca_layer(support_feat, query_feat)

        acc = accuracy(logits, query_target.reshape(-1))
        return logits, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        (
            ep_images,
            ep_global_targets,
            g_images,
            g_global_targets,
        ) = batch  # RENet uses both episode and general dataloaders
        ep_images = ep_images.to(self.device)  # ew(qs) c h w
        g_images = g_images.to(self.device)  # b c h w
        ep_global_targets = ep_global_targets.to(self.device)
        g_global_targets = g_global_targets.to(self.device)
        ep_global_targets_qry = ep_global_targets[
            ..., : self.query_num
        ]  # [e x w x (q+s)] -> [e x w x q]

        # extract features
        ep_feat = self.encode(ep_images)
        g_feat = self.encode(g_images)  # [128, 640, 5, 5]

        # CCA for ep_images
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            ep_feat, mode=2
        )
        # ws c h w ; wq c h w
        _, _, c, h, w = support_feat.shape
        support_feat = support_feat.reshape([-1, c, h, w])
        query_feat = query_feat.reshape([-1, c, h, w])
        logits, qry_pooled = self.cca_layer(support_feat, query_feat)
        abs_logits = self.fc(qry_pooled)
        epi_loss = self.loss_func(logits, query_target.reshape(-1))
        abs_loss = self.loss_func(abs_logits, ep_global_targets_qry.reshape(-1))

        # FC for g_images
        g_feat = g_feat.mean(dim=[-1, -2])
        logits_aux = self.fc(g_feat)
        aux_loss = self.loss_func(logits_aux, g_global_targets.reshape(-1))
        aux_loss = aux_loss + abs_loss

        loss = self.lambda_epi * epi_loss + aux_loss

        acc = accuracy(logits, query_target.reshape(-1))

        return logits, acc, loss
