import torch
import torch.nn as nn

from core.utils import accuracy
from .meta_model import MetaModel


class Predictor(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):
        super(Predictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.ELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ELU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class VERSA_HEAD(nn.Module):
    def __init__(self, way_num, sample_num):
        super(VERSA_HEAD, self).__init__()
        self.way_num = way_num
        self.sample_num = sample_num
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self, query_feat, query_targets, weight_mean, weight_logvar, bias_mean, bias_logvar):
        query_targets = query_targets.contiguous().reshape(-1)
        episode_size = query_feat.size(0)
        logits_mean_query = torch.matmul(query_feat, weight_mean) + bias_mean
        logits_log_var_query = torch.log(
            torch.matmul(query_feat ** 2, torch.exp(weight_logvar)) + torch.exp(bias_logvar))
        logits_sample_query = self.sample_normal(logits_mean_query, logits_log_var_query,
                                                 self.sample_num).contiguous().reshape(-1, self.way_num)

        query_label_tiled = query_targets.repeat(self.sample_num)
        loss = -self.loss_func(logits_sample_query, query_label_tiled)
        # FIXME
        loss = loss.contiguous().reshape(episode_size, self.sample_num, -1).permute([1, 0, 2]).contiguous().reshape(
            self.sample_num, -1)
        task_score = torch.logsumexp(loss, dim=0) - torch.log(
            torch.as_tensor(self.sample_num, dtype=torch.float).to(query_feat.device))
        # loss = -torch.mean(task_score, dim=0)
        logits_sample_query = logits_sample_query.contiguous().reshape(self.sample_num, -1, self.way_num)
        averaged_predictions = torch.logsumexp(logits_sample_query, dim=0) - torch.log(
            torch.as_tensor(self.sample_num, dtype=torch.float).to(query_feat.device))
        return averaged_predictions, task_score

    def sample_normal(self, mu, log_variance, num_samples):
        shape = torch.cat([torch.as_tensor([num_samples]), torch.as_tensor(mu.size())])
        eps = torch.randn(shape.cpu().numpy().tolist()).to(log_variance.device)
        return mu + eps * torch.sqrt(torch.exp(log_variance))


class VERSA(MetaModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim, hid_dim, sample_num=10):
        super(VERSA, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.sample_num = sample_num
        self.weight_means = Predictor(self.feat_dim, self.hid_dim, self.hid_dim)
        self.weight_logvars = Predictor(self.feat_dim, self.hid_dim, self.hid_dim)
        self.bias_means = Predictor(self.feat_dim, self.hid_dim, 1)
        self.bias_logvars = Predictor(self.feat_dim, self.hid_dim, 1)

        # self.head = VERSA_HEAD(way_num, sample_num)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def set_forward(self, batch, ):
        images, _ = batch
        images = images.to(self.device)
        images.required_grad = True

        feat = self.model_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(feat, mode=1)
        episode_size = support_targets.size(0)
        query_targets = query_targets.contiguous().reshape(episode_size, -1)

        class_feat = torch.mean(support_feat.contiguous().reshape(episode_size, self.way_num, self.shot_num, -1),
                                dim=2, keepdim=False)

        weight_means = self.weight_means(class_feat).permute((0, 2, 1))
        weight_logvars = self.weight_logvars(class_feat).permute((0, 2, 1))
        bias_means = self.bias_means(class_feat).permute((0, 2, 1))
        bias_logvars = self.bias_logvars(class_feat).permute((0, 2, 1))

        prediction_list = []
        for i in range(episode_size):
            episode_query_feat = query_feat[i]
            weight_mean = weight_means[i]
            weight_logvar = weight_logvars[i]
            bias_mean = bias_means[i]
            bias_logvar = bias_logvars[i]

            logits_mean_query = torch.matmul(episode_query_feat, weight_mean) + bias_mean
            logits_log_var_query = torch.log(
                torch.matmul(episode_query_feat ** 2, torch.exp(weight_logvar)) + torch.exp(bias_logvar))
            logits_sample_query = self.sample_normal(logits_mean_query, logits_log_var_query,
                                                     self.sample_num).contiguous().reshape(-1, self.way_num)

            logits_sample_query = logits_sample_query.contiguous().reshape(self.sample_num, -1, self.way_num)
            averaged_predictions = torch.logsumexp(logits_sample_query, dim=0) - torch.log(
                torch.as_tensor(self.sample_num, dtype=torch.float).to(episode_query_feat.device))
            prediction_list.append(averaged_predictions)

        averaged_predictions = torch.cat(prediction_list, dim=0)
        prec1, _ = accuracy(averaged_predictions, query_targets.reshape(-1), topk=(1, 3))
        return averaged_predictions, prec1

    def set_forward_loss(self, batch, ):
        images, _ = batch
        images = images.to(self.device)
        images.required_grad = True

        feat = self.model_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(feat, mode=1)
        episode_size = support_targets.size(0)
        query_targets = query_targets.contiguous().reshape(episode_size, -1)

        class_feat = torch.mean(support_feat.contiguous().reshape(episode_size, self.way_num, self.shot_num, -1),
                                dim=2, keepdim=False)

        weight_means = self.weight_means(class_feat).permute((0, 2, 1))
        weight_logvars = self.weight_logvars(class_feat).permute((0, 2, 1))
        bias_means = self.bias_means(class_feat).permute((0, 2, 1))
        bias_logvars = self.bias_logvars(class_feat).permute((0, 2, 1))

        prediction_list = []
        task_score_list = []
        for i in range(episode_size):
            episode_query_feat = query_feat[i]
            episode_query_targets = query_targets[i]
            weight_mean = weight_means[i]
            weight_logvar = weight_logvars[i]
            bias_mean = bias_means[i]
            bias_logvar = bias_logvars[i]

            logits_mean_query = torch.matmul(episode_query_feat, weight_mean) + bias_mean
            logits_log_var_query = torch.log(
                torch.matmul(episode_query_feat ** 2, torch.exp(weight_logvar)) + torch.exp(bias_logvar))
            logits_sample_query = self.sample_normal(logits_mean_query, logits_log_var_query,
                                                     self.sample_num).contiguous().reshape(-1, self.way_num)

            query_label_tiled = episode_query_targets.repeat(self.sample_num)
            loss = -self.loss_func(logits_sample_query, query_label_tiled)
            loss = loss.contiguous().reshape(self.sample_num, -1)
            task_score = torch.logsumexp(loss, dim=0) - torch.log(
                torch.as_tensor(self.sample_num, dtype=torch.float).to(episode_query_feat.device))
            logits_sample_query = logits_sample_query.contiguous().reshape(self.sample_num, -1, self.way_num)
            averaged_predictions = torch.logsumexp(logits_sample_query, dim=0) - torch.log(
                torch.as_tensor(self.sample_num, dtype=torch.float).to(episode_query_feat.device))
            prediction_list.append(averaged_predictions)
            task_score_list.append(task_score)

        averaged_predictions = torch.cat(prediction_list, dim=0)
        prec1, _ = accuracy(averaged_predictions, query_targets.reshape(-1), topk=(1, 3))
        task_score = torch.cat(task_score_list, dim=0)
        loss = -torch.mean(task_score, dim=0)
        return averaged_predictions, prec1, loss

    def train_loop(self, *args, **kwargs):
        pass

    def test_loop(self, *args, **kwargs):
        pass

    def set_forward_adaptation(self, *args, **kwargs):
        pass

    def sample_normal(self, mu, log_variance, num_samples):
        shape = torch.cat([torch.as_tensor([num_samples]), torch.as_tensor(mu.size())])
        eps = torch.randn(shape.cpu().numpy().tolist()).to(log_variance.device)
        return mu + eps * torch.sqrt(torch.exp(log_variance))
