import torch
from torch import nn
import math

from core.utils import accuracy
from .meta_model import MetaModel


def sample(weights, size):
    mean, var = weights[:, :, :size], weights[:, :, size:]
    z = torch.normal(0.0, 1.0, mean.size()).to(weights.device)
    return mean + var * z


def cal_log_prob(x, mean, var):
    eps = 1e-20
    log_unnormalized = - 0.5 * ((x - mean) / (var + eps)) ** 2
    # log_normalization = torch.log(var + eps) + 0.5 * math.log(2 * math.pi)
    log_normalization = torch.log(var + eps) + 0.5 * torch.log(2 * torch.tensor(math.pi))
    return log_unnormalized - log_normalization


def cal_kl_div(latents, mean, var):
    return torch.mean(
        cal_log_prob(latents, mean, var) - cal_log_prob(latents, torch.zeros(mean.size()).to(latents.device),
                                                        torch.ones(var.size()).to(latents.device)))


def orthogonality(weight):
    w2 = torch.mm(weight, weight.transpose(0, 1))
    wn = torch.norm(weight, dim=1, keepdim=True) + 1e-20
    correlation_matrix = w2 / torch.mm(wn, wn.transpose(0, 1))
    assert correlation_matrix.size(0) == correlation_matrix.size(1)
    I = torch.eye(correlation_matrix.size(0)).to(weight.device)
    return torch.mean((correlation_matrix - I) ** 2)


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
        x = x.reshape(episode_size, self.way_num, self.way_num * self.shot_num * self.shot_num, -1)
        x = torch.mean(x, dim=2)

        latents = sample(x, self.hid_dim)
        mean, var = x[:, :, :self.hid_dim], x[:, :, self.hid_dim:]
        kl_div = cal_kl_div(latents, mean, var)

        return latents, kl_div


class Decoder(nn.Module):
    def __init__(self, feat_dim, hid_dim):
        super(Decoder, self).__init__()
        self.decoder_func = nn.Linear(hid_dim, 2 * feat_dim)

    def forward(self, x):
        return self.decoder_func(x)


#
# class LEO_HEAD(nn.Module):
#     def __init__(self, way_num, shot_num, feat_dim, hid_dim):
#         super(LEO_HEAD, self).__init__()
#         self.way_num = way_num
#         self.shot_num = shot_num
#         self.feat_dim = feat_dim
#         self.hid_dim = hid_dim
#         self.encoder_func = nn.Linear(self.feat_dim, self.hid_dim)
#         self.relation_net = nn.Sequential(
#             nn.Linear(2 * self.hid_dim, 2 * self.hid_dim, bias=False),
#             nn.ReLU(),
#             nn.Linear(2 * self.hid_dim, 2 * self.hid_dim, bias=False),
#             nn.ReLU(),
#             nn.Linear(2 * self.hid_dim, 2 * self.hid_dim, bias=False),
#             nn.ReLU()
#         )
#         self.decoder_func = nn.Linear(self.hid_dim, 2 * self.feat_dim)
#
#     def encoder(self, x):
#         return self.encoder_func(x)
#
#     def decoder(self, x):
#         return self.decoder_func(x)


class LEO(MetaModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim, hid_dim, inner_optim,
                 inner_train_iter=10, finetune_iter=10, kl_weight=1, encoder_penalty_weight=1,
                 orthogonality_penalty_weight=1):
        super(LEO, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.encoder = Encoder(way_num, shot_num, feat_dim, hid_dim)
        self.decoder = Decoder(feat_dim, hid_dim)
        self.inner_optim = inner_optim
        self.inner_train_iter = inner_train_iter
        self.finetune_iter = finetune_iter
        self.kl_weight = kl_weight
        self.encoder_penalty_weight = encoder_penalty_weight
        self.orthogonality_penalty_weight = orthogonality_penalty_weight

        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch, ):
        images, global_targets = batch
        images = images.to(self.device)

        with torch.no_grad():
            emb = self.model_func(images)
        emb_support, emb_query, support_targets, query_targets = self.split_by_episode(emb, mode=1)
        episode_size = emb_support.size(0)

        latents, kl_div, encoder_penalty = self.train_loop(emb_support, support_targets, episode_size)

        classifier_weights = self.decoder(latents)
        classifier_weights = sample(classifier_weights, self.feat_dim)
        classifier_weights = classifier_weights.permute([0, 2, 1])

        classifier_weights = self.finetune(classifier_weights, emb_support, support_targets)

        output = torch.bmm(emb_query, classifier_weights)
        output = output.contiguous().reshape(-1, self.way_num)

        prec1, _ = accuracy(output, query_targets.contiguous().reshape(-1), topk=(1, 3))
        return output, prec1

    def set_forward_loss(self, batch, ):
        images, global_targets = batch
        images = images.to(self.device)

        with torch.no_grad():
            emb = self.model_func(images)
        emb_support, emb_query, support_targets, query_targets = self.split_by_episode(emb, mode=1)
        episode_size = emb_support.size(0)

        latents, kl_div, encoder_penalty = self.train_loop(emb_support, support_targets, episode_size)

        classifier_weights = self.decoder(latents)
        classifier_weights = sample(classifier_weights, self.feat_dim)
        classifier_weights = classifier_weights.permute([0, 2, 1])

        classifier_weights = self.finetune(classifier_weights, emb_support, support_targets)

        output = torch.bmm(emb_query, classifier_weights)
        output = output.contiguous().reshape(-1, self.way_num)
        pred_loss = self.loss_func(output, query_targets.contiguous().reshape(-1))

        orthogonality_penalty = orthogonality(list(self.decoder.parameters())[0])

        total_loss = pred_loss + self.kl_weight * kl_div + self.encoder_penalty_weight * encoder_penalty + self.orthogonality_penalty_weight * orthogonality_penalty
        prec1, _ = accuracy(output, query_targets.contiguous().reshape(-1), topk=(1, 3))
        return output, prec1, total_loss

    def train_loop(self, emb_support, support_targets, episode_size):
        latents, kl_div = self.encoder(emb_support)
        latents_init = latents
        for i in range(self.inner_train_iter):
            latents.retain_grad()
            classifier_weight = self.decoder(latents)
            classifier_weight = sample(classifier_weight, self.feat_dim)
            classifier_weight = classifier_weight.permute([0, 2, 1])
            output = torch.bmm(emb_support, classifier_weight)
            output = output.contiguous().reshape(-1, self.way_num)
            targets = support_targets.contiguous().reshape(-1)
            loss = self.loss_func(output, targets)

            loss.backward(retain_graph=True)

            latents = latents - self.inner_optim['inner_lr'] * latents.grad

        encoder_penalty = torch.mean((latents_init - latents) ** 2)
        return latents, kl_div, encoder_penalty

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError

    def finetune(self, classifier_weights, emb_support, support_targets):
        classifier_weights.retain_grad()
        output = torch.bmm(emb_support, classifier_weights)
        output = output.contiguous().reshape(-1, self.way_num)
        targets = support_targets.contiguous().reshape(-1)
        pred_loss = self.loss_func(output, targets)

        for j in range(self.finetune_iter):
            pred_loss.backward(retain_graph=True)
            classifier_weights = classifier_weights - self.inner_optim['finetune_lr'] * classifier_weights.grad
            classifier_weights.retain_grad()

            output = torch.bmm(emb_support, classifier_weights)
            output = output.contiguous().reshape(-1, self.way_num)
            targets = support_targets.contiguous().reshape(-1)
            pred_loss = self.loss_func(output, targets)

        return classifier_weights
