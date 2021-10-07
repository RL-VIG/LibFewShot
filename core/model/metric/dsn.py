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
  url       = {https://openaccess.thecvf.com/content\_CVPR\_2020/html/Simon\_Adaptive\_Subspaces\_for\_Few-Shot\_Learning\_CVPR\_2020\_paper.html},
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

    def forward(self, query_feat, support_feat, way_num, shot_num, normalize=True):
        e, ws, d = support_feat.size()

        support_feat = support_feat.reshape(e, way_num, shot_num, -1)

        UU, _, _ = torch.svd(support_feat.permute(0, 1, 3, 2))
        subspace = UU[:, :, :, :shot_num-1].permute(0,1,3,2)

        projection = subspace.permute(0,1,3,2).matmul(subspace.matmul(query_feat.permute(0, 2, 1))).permute(0, 1, 3, 2)

        dist = torch.sum((query_feat - projection) ** 2, dim=-1).permute(0, 2, 1)

        logits = -dist
        if normalize:
            logits /= d

        return logits

class DSN(MetricModel):
    def __init__(self, eps=0.1, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.dsn_layer = DSNLayer()

        self.eps = eps

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))

        if self.shot_num == 1:
            # B, C, H, W
            support_images, query_images, support_target, query_target = self.split_by_episode(image, mode=2)

            # change shot_num
            self.shot_num = 2
            # eouside, way, shot, channel, height, width
            flipped_images = torch.flip(support_images, dims=[-1])
            support_images = torch.stack([support_images, flipped_images], dim=2)

            e, w, s, c, h_, w_ = support_images.size()
            support_images = support_images.reshape(e*w*s, c, h_, w_)
            query_images = query_images.reshape(-1, c, h_, w_)

            all_images = torch.cat([support_images, query_images], dim=0)

            all_feat = self.emb_func(all_images)
            support_feat = all_feat[:e*w*s]
            query_feat = all_feat[e*w*s:]
        else:
            feat = self.emb_func(image)

            support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)

        output = self.dsn_layer(query_feat, support_feat, self.way_num, self.shot_num).view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.view(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)

        smoothed_query_target = F.one_hot(query_target.view(-1), self.way_num)
        smoothed_query_target = smoothed_query_target * (1 - self.eps) + (1 - smoothed_query_target) * self.eps / (self.way_num - 1)

        output = self.dsn_layer(query_feat, support_feat, self.way_num, self.shot_num).view(episode_size * self.way_num * self.query_num, self.way_num)
        # loss = self.loss_fn(output, query_target.view(-1))
        log_prb = F.log_softmax(output.reshape(-1, self.way_num))
        loss = -(smoothed_query_target * log_prb).sum(dim=1).mean()
        acc = accuracy(output, query_target.view(-1))

        return output, acc, loss