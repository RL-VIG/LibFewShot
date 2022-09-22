# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/SimonKNH20,
  author    = {Christian Simon and
               Piotr Koniusz and
               Richard Nock and
               Mehrtash Harandi},
  title     = {Adaptive Subspaces for Few-Shot Learning},
  booktitle = {2020 {IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2020, Seattle, WA, USA, June 13-19, 2020},
  pages     = {4135--4144},
  publisher = {Computer Vision Foundation / {IEEE}},
  year      = {2020},
  url       = {https://openaccess.thecvf.com/content_CVPR_2020/htmlSimon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.html},
  doi       = {10.1109/CVPR42600.2020.00419},
  timestamp = {Tue, 31 Aug 2021 14:00:04 +0200},
  biburl    = {https://dblp.org/rec/conf/cvpr/SimonKNH20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
https://openaccess.thecvf.com/content_CVPR_2020/html/Simon_Adaptive_Subspaces_for_Few-Shot_Learning_CVPR_2020_paper.html

Adapted from https://github.com/chrysts/dsn_fewshot.
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class DSNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        normalize=True,
        discriminative=False,
    ):
        e, ws, d = support_feat.size()

        support_feat = support_feat.reshape(e, way_num, shot_num, -1)
        # add way dim
        query_feat = query_feat.unsqueeze(1)

        try:
            # faster but need pytorch>=1.8.0
            UU, _, _ = torch.linalg.svd(
                support_feat.permute(0, 1, 3, 2).double()
            )  # [episode_size, way_num, dim, dim]
        except AttributeError:
            UU, _, _ = torch.svd(
                support_feat.permute(0, 1, 3, 2).double()
            )  # [episode_size, way_num, dim, dim]
        UU = UU.float()
        subspace = UU[:, :, :, : shot_num - 1].permute(
            0, 1, 3, 2
        )  # [episode_size, way_num, subspace_dim, dim]

        projection = (
            subspace.permute(0, 1, 3, 2)
            .matmul(subspace.matmul(query_feat.permute(0, 1, 3, 2)))
            .permute(0, 1, 3, 2)
        )  # [episode_size, way_num, shot_num, dim]

        dist = torch.sum((query_feat - projection) ** 2, dim=-1).permute(0, 2, 1)

        logits = -dist
        if normalize:
            logits /= d

        disc_loss = 0
        if discriminative:
            # P is [dim, subspace_dim] while subspace here is [subspace_dim, dim]
            subspace_metric = torch.norm(
                torch.matmul(
                    subspace.unsqueeze(1), subspace.unsqueeze(2).transpose(-2, -1)
                ),
                p="fro",
                dim=[-2, -1],
            )  # [episode_size, way_num, way_num]
            mask = torch.eye(way_num).bool()
            subspace_metric = subspace_metric[:, ~mask]
            disc_loss = torch.sum(subspace_metric**2)

        return logits, disc_loss


class DSN(MetricModel):
    def __init__(
        self, eps=0.1, enable_scale=True, lamb=0.03, discriminative=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.dsn_layer = DSNLayer()

        self.lamb = lamb
        self.discriminative = discriminative
        self.eps = eps
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

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

        if self.shot_num == 1:
            _, c, h, w = image.size()
            support_image = image.reshape(episode_size, self.way_num, -1, c, h, w)[
                :, :, : self.shot_num, :, :, :
            ]
            query_image = image.reshape(episode_size, self.way_num, -1, c, h, w)[
                :, :, self.shot_num :, :, :, :
            ]

            support_image = (
                torch.stack([support_image, torch.flip(support_image, dims=[3])])
                .permute(1, 2, 3, 0, 4, 5, 6)
                .reshape(episode_size, self.way_num, -1, c, h, w)
            )

            image = torch.cat([support_image, query_image], dim=2).reshape(-1, c, h, w)

            self.shot_num = 2

            feat = self.emb_func(image)

            (
                support_feat,
                query_feat,
                support_target,
                query_target,
            ) = self.split_by_episode(feat, mode=1)

            smoothed_query_target = F.one_hot(query_target.reshape(-1), self.way_num)
            smoothed_query_target = smoothed_query_target * (1 - self.eps) + (
                1 - smoothed_query_target
            ) * self.eps / (self.way_num - 1)

            output, _ = self.dsn_layer(
                query_feat, support_feat, self.way_num, self.shot_num
            )

            self.shot_num = 1
        else:
            feat = self.emb_func(image)

            (
                support_feat,
                query_feat,
                support_target,
                query_target,
            ) = self.split_by_episode(feat, mode=1)

            output, _ = self.dsn_layer(
                query_feat, support_feat, self.way_num, self.shot_num
            )

        output = output.reshape(
            episode_size * self.way_num * self.query_num, self.way_num
        )
        output = output * self.scale if self.enable_scale else output
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

        if self.shot_num == 1:
            _, c, h, w = image.size()
            support_image = image.reshape(episode_size, self.way_num, -1, c, h, w)[
                :, :, : self.shot_num, :, :, :
            ]
            query_image = image.reshape(episode_size, self.way_num, -1, c, h, w)[
                :, :, self.shot_num :, :, :, :
            ]

            support_image = (
                torch.stack([support_image, torch.flip(support_image, dims=[3])])
                .permute(1, 2, 3, 0, 4, 5, 6)
                .reshape(episode_size, self.way_num, -1, c, h, w)
            )

            image = torch.cat([support_image, query_image], dim=2).reshape(-1, c, h, w)

            self.shot_num = 2

            feat = self.emb_func(image)

            (
                support_feat,
                query_feat,
                support_target,
                query_target,
            ) = self.split_by_episode(feat, mode=1)

            smoothed_query_target = F.one_hot(query_target.reshape(-1), self.way_num)
            smoothed_query_target = smoothed_query_target * (1 - self.eps) + (
                1 - smoothed_query_target
            ) * self.eps / (self.way_num - 1)

            output, disc_loss = self.dsn_layer(
                query_feat,
                support_feat,
                self.way_num,
                self.shot_num,
                discriminative=self.discriminative,
            )

            self.shot_num = 1
        else:
            feat = self.emb_func(image)

            (
                support_feat,
                query_feat,
                support_target,
                query_target,
            ) = self.split_by_episode(feat, mode=1)

            smoothed_query_target = F.one_hot(query_target.reshape(-1), self.way_num)
            smoothed_query_target = smoothed_query_target * (1 - self.eps) + (
                1 - smoothed_query_target
            ) * self.eps / (self.way_num - 1)

            output, disc_loss = self.dsn_layer(
                query_feat,
                support_feat,
                self.way_num,
                self.shot_num,
                discriminative=self.discriminative,
            )

        output = output.reshape(
            episode_size * self.way_num * self.query_num, self.way_num
        )
        output = output * self.scale if self.enable_scale else output
        # loss = self.loss_fn(output, query_target.reshape(-1))
        log_prb = F.log_softmax(output, dim=-1)
        loss = -(smoothed_query_target * log_prb).sum(dim=1).mean()

        loss += self.lamb * disc_loss
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss
