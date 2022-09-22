# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/iclr/RusuRSVPOH19,
  author    = {Andrei A. Rusu and
               Dushyant Rao and
               Jakub Sygnowski and
               Oriol Vinyals and
               Razvan Pascanu and
               Simon Osindero and
               Raia Hadsell},
  title     = {Meta-Learning with Latent Embedding Optimization},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=BJgklhAcK7}
}
https://arxiv.org/abs/1807.05960

Adapted from https://github.com/deepmind/leo.
"""
import torch
from torch import nn
import math

from core.utils import accuracy
from .meta_model import MetaModel


def sample(weight, size):
    mean, var = weight[:, :, :size], weight[:, :, size:]
    z = torch.normal(0.0, 1.0, mean.size()).to(weight.device)
    return mean + var * z


def cal_log_prob(x, mean, var):
    eps = 1e-20
    log_unnormalized = -0.5 * ((x - mean) / (var + eps)) ** 2
    # log_normalization = torch.log(var + eps) + 0.5 * math.log(2 * math.pi)
    log_normalization = torch.log(var + eps) + 0.5 * torch.log(
        2 * torch.tensor(math.pi)
    )
    return log_unnormalized - log_normalization


def cal_kl_div(latent, mean, var):
    return torch.mean(
        cal_log_prob(latent, mean, var)
        - cal_log_prob(
            latent,
            torch.zeros(mean.size()).to(latent.device),
            torch.ones(var.size()).to(latent.device),
        )
    )


def orthogonality(weight):
    w2 = torch.mm(weight, weight.transpose(0, 1))
    wn = torch.norm(weight, dim=1, keepdim=True) + 1e-20
    correlation_matrix = w2 / torch.mm(wn, wn.transpose(0, 1))
    assert correlation_matrix.size(0) == correlation_matrix.size(
        1
    ), "correlation_matrix is not square, correlation_matrix.shape is {}".format(
        correlation_matrix.shape
    )
    identity_matrix = torch.eye(correlation_matrix.size(0)).to(weight.device)
    return torch.mean((correlation_matrix - identity_matrix) ** 2)


class Encoder(nn.Module):
    def __init__(self, way_num, shot_num, feat_dim, hid_dim, drop_prob=0.0):
        super(Encoder, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.encoder_func = nn.Linear(feat_dim, hid_dim)
        self.relation_net = nn.Sequential(
            nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2 * hid_dim, 2 * hid_dim, bias=False),
            nn.ReLU(),
        )
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.drop_out(x)
        out = self.encoder_func(x)
        episode_size = out.size(0)
        out = out.contiguous().reshape(episode_size, self.way_num, self.shot_num, -1)

        # for relation net
        t1 = torch.repeat_interleave(out, self.shot_num, dim=2)
        t1 = torch.repeat_interleave(t1, self.way_num, dim=1)
        t2 = out.repeat((1, self.way_num, self.shot_num, 1))
        x = torch.cat((t1, t2), dim=-1)

        x = self.relation_net(x)
        x = x.reshape(
            episode_size,
            self.way_num,
            self.way_num * self.shot_num * self.shot_num,
            -1,
        )
        x = torch.mean(x, dim=2)

        latent = sample(x, self.hid_dim)
        mean, var = x[:, :, : self.hid_dim], x[:, :, self.hid_dim :]
        kl_div = cal_kl_div(latent, mean, var)

        return latent, kl_div


class Decoder(nn.Module):
    def __init__(self, feat_dim, hid_dim):
        super(Decoder, self).__init__()
        self.decoder_func = nn.Linear(hid_dim, 2 * feat_dim)

    def forward(self, x):
        return self.decoder_func(x)


class LEO(MetaModel):
    def __init__(
        self,
        inner_para,
        feat_dim,
        hid_dim,
        kl_weight,
        encoder_penalty_weight,
        orthogonality_penalty_weight,
        **kwargs
    ):
        super(LEO, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.encoder = Encoder(self.way_num, self.shot_num, feat_dim, hid_dim)
        self.decoder = Decoder(feat_dim, hid_dim)
        self.inner_para = inner_para
        self.kl_weight = kl_weight
        self.encoder_penalty_weight = encoder_penalty_weight
        self.orthogonality_penalty_weight = orthogonality_penalty_weight

        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)

        latents, kl_div, encoder_penalty = self.set_forward_adaptation(
            support_feat, support_target, episode_size
        )

        leo_weight = self.decoder(latents)
        leo_weight = sample(leo_weight, self.feat_dim)
        leo_weight = leo_weight.permute([0, 2, 1])

        leo_weight = self.finetune(leo_weight, support_feat, support_target)

        output = torch.bmm(query_feat, leo_weight)
        output = output.contiguous().reshape(-1, self.way_num)

        acc = accuracy(output, query_target.contiguous().reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)

        latent, kl_div, encoder_penalty = self.set_forward_adaptation(
            support_feat, support_target, episode_size
        )

        classifier_weight = self.decoder(latent)
        classifier_weight = sample(classifier_weight, self.feat_dim)
        classifier_weight = classifier_weight.permute([0, 2, 1])

        classifier_weight = self.finetune(
            classifier_weight, support_feat, support_target
        )

        output = torch.bmm(query_feat, classifier_weight)
        output = output.contiguous().reshape(-1, self.way_num)
        pred_loss = self.loss_func(output, query_target.contiguous().reshape(-1))

        orthogonality_penalty = orthogonality(list(self.decoder.parameters())[0])

        total_loss = (
            pred_loss
            + self.kl_weight * kl_div
            + self.encoder_penalty_weight * encoder_penalty
            + self.orthogonality_penalty_weight * orthogonality_penalty
        )
        acc = accuracy(output, query_target.contiguous().reshape(-1))
        return output, acc, total_loss

    def set_forward_adaptation(self, emb_support, support_target, episode_size):
        latent, kl_div = self.encoder(emb_support)
        latent_init = latent
        for i in range(self.inner_para["iter"]):
            latent.retain_grad()
            classifier_weight = self.decoder(latent)
            classifier_weight = sample(classifier_weight, self.feat_dim)
            classifier_weight = classifier_weight.permute([0, 2, 1])
            output = torch.bmm(emb_support, classifier_weight)
            output = output.contiguous().reshape(-1, self.way_num)
            targets = support_target.contiguous().reshape(-1)
            loss = self.loss_func(output, targets)

            loss.backward(retain_graph=True)

            latent = latent - self.inner_para["lr"] * latent.grad

        encoder_penalty = torch.mean((latent_init - latent) ** 2)
        return latent, kl_div, encoder_penalty

    def finetune(self, classifier_weight, emb_support, support_target):
        classifier_weight.retain_grad()
        output = torch.bmm(emb_support, classifier_weight)
        output = output.contiguous().reshape(-1, self.way_num)
        target = support_target.contiguous().reshape(-1)
        pred_loss = self.loss_func(output, target)

        for j in range(self.inner_para["finetune_iter"]):
            pred_loss.backward(retain_graph=True)
            classifier_weight = (
                classifier_weight
                - self.inner_para["finetune_lr"] * classifier_weight.grad
            )
            classifier_weight.retain_grad()

            output = torch.bmm(emb_support, classifier_weight)
            output = output.contiguous().reshape(-1, self.way_num)
            targets = support_target.contiguous().reshape(-1)
            pred_loss = self.loss_func(output, targets)

        return classifier_weight
