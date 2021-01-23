import torch
import torch.nn as nn

from core.utils import accuracy
from .meta_model import MetaModel


class Classifier(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):
        super(Classifier, self).__init__()
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


class VERSA(MetaModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim, hid_dim, sample_num=10):
        super(VERSA, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.sample_num = sample_num
        self.weight_means = Classifier(self.feat_dim, self.hid_dim, self.hid_dim)
        self.weight_logvars = Classifier(self.feat_dim, self.hid_dim, self.hid_dim)
        self.bias_means = Classifier(self.feat_dim, self.hid_dim, 1)
        self.bias_logvars = Classifier(self.feat_dim, self.hid_dim, 1)

        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch, ):
        images, _ = batch
        images = images.to(self.device)

        feat = self.model_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(feat, mode=1)
        episode_size = support_targets.size(0)

        output_list = []
        for i in range(episode_size):
            episode_support_feat = support_feat[i]
            episode_query_feat = query_feat[i]
            class_feat = torch.mean(episode_support_feat.contiguous().reshape(self.way_num, self.shot_num, -1),
                                    dim=1, keepdim=False)

            weight_means = self.weight_means(class_feat).T
            weight_logvars = self.weight_logvars(class_feat).T
            bias_means = self.bias_means(class_feat).T
            bias_logvars = self.bias_logvars(class_feat).T

            logits_mean_query = torch.matmul(episode_query_feat, weight_means) + bias_means
            logits_log_var_query = torch.log(
                torch.matmul(episode_query_feat ** 2, torch.exp(weight_logvars)) + torch.exp(bias_logvars))
            logits_sample_query = self.sample_normal(logits_mean_query, logits_log_var_query, self.sample_num).reshape(
                -1, self.way_num)
            # query_label_tiled = query_targets.repeat(self.sample_num)
            output_list.append(logits_sample_query)

        query_label_tiled = query_targets.repeat(episode_size * self.sample_num)
        output = torch.cat(output_list, dim=0)
        prec1, _ = accuracy(output, query_label_tiled, topk=(1, 3))
        return output, prec1

    def set_forward_loss(self, batch, ):
        images, _ = batch
        images = images.to(self.device)

        feat = self.model_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(feat, mode=1)
        episode_size = support_targets.size(0)

        output_list = []
        for i in range(episode_size):
            episode_support_feat = support_feat[i]
            episode_query_feat = query_feat[i]
            class_feat = torch.mean(episode_support_feat.contiguous().reshape(self.way_num, self.shot_num, -1),
                                    dim=1, keepdim=False)

            weight_means = self.weight_means(class_feat).T
            weight_logvars = self.weight_logvars(class_feat).T
            bias_means = self.bias_means(class_feat).T
            bias_logvars = self.bias_logvars(class_feat).T

            logits_mean_query = torch.matmul(episode_query_feat, weight_means) + bias_means
            logits_log_var_query = torch.log(
                torch.matmul(episode_query_feat ** 2, torch.exp(weight_logvars)) + torch.exp(bias_logvars))
            logits_sample_query = self.sample_normal(logits_mean_query, logits_log_var_query, self.sample_num).reshape(
                -1, self.way_num)
            # query_label_tiled = query_targets.repeat(self.sample_num)
            output_list.append(logits_sample_query)

        query_label_tiled = query_targets.repeat(episode_size * self.sample_num)
        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_label_tiled)
        prec1, _ = accuracy(output, query_label_tiled, topk=(1, 3))
        return output, prec1, loss

    def train_loop(self, *args, **kwargs):
        pass

    def test_loop(self, *args, **kwargs):
        pass

    def set_forward_adaptation(self, *args, **kwargs):
        pass

    def sample_normal(self, mu, log_variance, num_samples):
        shape = torch.cat([torch.as_tensor([num_samples]), torch.as_tensor(mu.size())])
        eps = torch.randn(shape.cpu().numpy().tolist()).to(self.device)
        return mu + eps * torch.sqrt(torch.exp(log_variance))
