# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/GordonBBNT19,
  author    = {Jonathan Gordon and
               John Bronskill and
               Matthias Bauer and
               Sebastian Nowozin and
               Richard E. Turner},
  title     = {Meta-Learning Probabilistic Inference for Prediction},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=HkxStoC5F7}
}
https://openreview.net/forum?id=HkxStoC5F7

Adapted from https://github.com/Gordonjo/versa.
"""
import torch
import torch.nn as nn

from core.utils import accuracy
from .meta_model import MetaModel


class Predictor(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):
        super(Predictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.ELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ELU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class VERSALayer(nn.Module):
    def __init__(self, sample_num):
        super(VERSALayer, self).__init__()
        self.sample_num = sample_num
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        way_num,
        query_feat,
        query_target,
        weight_mean,
        weight_logvar,
        bias_mean,
        bias_logvar,
    ):
        query_target = query_target.contiguous().reshape(-1)
        episode_size = query_feat.size(0)
        logits_mean_query = torch.matmul(query_feat, weight_mean) + bias_mean
        logits_log_var_query = torch.log(
            torch.matmul(query_feat**2, torch.exp(weight_logvar))
            + torch.exp(bias_logvar)
        )
        logits_sample_query = (
            self.sample_normal(logits_mean_query, logits_log_var_query, self.sample_num)
            .contiguous()
            .reshape(-1, way_num)
        )

        query_label_tiled = query_target.repeat(self.sample_num)
        loss = -self.loss_func(logits_sample_query, query_label_tiled)
        # FIXME nan
        loss = (
            loss.contiguous()
            .reshape(episode_size, self.sample_num, -1)
            .permute([1, 0, 2])
            .contiguous()
            .reshape(self.sample_num, -1)
        )
        task_score = torch.logsumexp(loss, dim=0) - torch.log(
            torch.as_tensor(self.sample_num, dtype=torch.float).to(query_feat.device)
        )
        # loss = -torch.mean(task_score, dim=0)
        logits_sample_query = logits_sample_query.contiguous().reshape(
            self.sample_num, -1, way_num
        )
        averaged_prediction = torch.logsumexp(logits_sample_query, dim=0) - torch.log(
            torch.as_tensor(self.sample_num, dtype=torch.float).to(query_feat.device)
        )
        return averaged_prediction, task_score

    def sample_normal(self, mu, log_variance, num_samples):
        shape = torch.cat([torch.as_tensor([num_samples]), torch.as_tensor(mu.size())])
        eps = torch.randn(shape.cpu().numpy().tolist()).to(log_variance.device)
        return mu + eps * torch.sqrt(torch.exp(log_variance))


class VERSA(MetaModel):
    def __init__(self, feat_dim, sample_num, d_theta=256, drop_rate=0.0, **kwargs):
        super(VERSA, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.sample_num = sample_num
        self.h = nn.Sequential(
            nn.Linear(feat_dim, d_theta),
            nn.BatchNorm1d(d_theta),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.weight_mean = Predictor(d_theta, d_theta, d_theta)
        self.weight_logvar = Predictor(d_theta, d_theta, d_theta)
        self.bias_mean = Predictor(d_theta, d_theta, 1)
        self.bias_logvar = Predictor(d_theta, d_theta, 1)
        self.head = VERSALayer(sample_num)

    @torch.no_grad()
    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        feat = self.h(feat)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)
        query_target = query_target.contiguous().reshape(episode_size, -1)

        class_feat = torch.mean(
            support_feat.contiguous().reshape(
                episode_size, self.way_num, self.shot_num, -1
            ),
            dim=2,
            keepdim=False,
        )

        weight_mean = self.weight_mean(class_feat).permute((0, 2, 1))
        weight_logvar = self.weight_logvar(class_feat).permute((0, 2, 1))
        bias_mean = self.bias_mean(class_feat).permute((0, 2, 1))
        bias_logvar = self.bias_logvar(class_feat).permute((0, 2, 1))

        output, _ = self.head(
            self.way_num,
            query_feat,
            query_target,
            weight_mean,
            weight_logvar,
            bias_mean,
            bias_logvar,
        )
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        feat = self.h(feat)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)
        query_target = query_target.contiguous().reshape(episode_size, -1)

        class_feat = torch.mean(
            support_feat.contiguous().reshape(
                episode_size, self.way_num, self.shot_num, -1
            ),
            dim=2,
            keepdim=False,
        )

        weight_mean = self.weight_mean(class_feat).permute((0, 2, 1))
        weight_logvar = self.weight_logvar(class_feat).permute((0, 2, 1))
        bias_mean = self.bias_mean(class_feat).permute((0, 2, 1))
        bias_logvar = self.bias_logvar(class_feat).permute((0, 2, 1))

        output, task_score = self.head(
            self.way_num,
            query_feat,
            query_target,
            weight_mean,
            weight_logvar,
            bias_mean,
            bias_logvar,
        )
        acc = accuracy(output, query_target.reshape(-1))
        task_score = self.drop_nan(task_score)
        loss = -torch.mean(task_score, dim=0)
        return output, acc, loss

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError

    def drop_nan(self, tensor):
        tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, 0), tensor)

        return tensor
