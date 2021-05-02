import torch
import torch.nn as nn

from core.utils import accuracy
from .meta_model import MetaModel

# adapted from offical tf codes: https://github.com/Gordonjo/versas

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


class VERSA_Layer(nn.Module):
    def __init__(self, train_way, sample_num):
        super(VERSA_Layer, self).__init__()
        self.train_way = train_way
        self.sample_num = sample_num
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self, query_feat, query_target, weight_mean, weight_logvar, bias_mean, bias_logvar):
        query_target = query_target.contiguous().reshape(-1)
        episode_size = query_feat.size(0)
        logits_mean_query = torch.matmul(query_feat, weight_mean) + bias_mean
        logits_log_var_query = torch.log(
            torch.matmul(query_feat ** 2, torch.exp(weight_logvar)) + torch.exp(bias_logvar))
        logits_sample_query = self.sample_normal(logits_mean_query, logits_log_var_query,
                                                 self.sample_num).contiguous().reshape(-1, self.train_way)

        query_label_tiled = query_target.repeat(self.sample_num)
        loss = -self.loss_func(logits_sample_query, query_label_tiled)
        # FIXME nan
        loss = loss.contiguous().reshape(episode_size, self.sample_num, -1).permute([1, 0, 2]).contiguous().reshape(
            self.sample_num, -1)
        task_score = torch.logsumexp(loss, dim=0) - torch.log(
            torch.as_tensor(self.sample_num, dtype=torch.float).to(query_feat.device))
        # loss = -torch.mean(task_score, dim=0)
        logits_sample_query = logits_sample_query.contiguous().reshape(self.sample_num, -1, self.train_way)
        averaged_prediction = torch.logsumexp(logits_sample_query, dim=0) - torch.log(
            torch.as_tensor(self.sample_num, dtype=torch.float).to(query_feat.device))
        return averaged_prediction, task_score

    def sample_normal(self, mu, log_variance, num_samples):
        shape = torch.cat([torch.as_tensor([num_samples]), torch.as_tensor(mu.size())])
        eps = torch.randn(shape.cpu().numpy().tolist()).to(log_variance.device)
        return mu + eps * torch.sqrt(torch.exp(log_variance))


class VERSA(MetaModel):
    def __init__(self, train_way, train_shot, train_query, emb_func, device, feat_dim, hid_dim, sample_num):
        super(VERSA, self).__init__(train_way, train_shot, train_query, emb_func, device)
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.sample_num = sample_num
        self.weight_mean = Predictor(self.feat_dim, self.hid_dim, self.feat_dim)
        self.weight_logvar = Predictor(self.feat_dim, self.hid_dim, self.feat_dim)
        self.bias_mean = Predictor(self.feat_dim, self.hid_dim, 1)
        self.bias_logvar = Predictor(self.feat_dim, self.hid_dim, 1)

        self.head = VERSA_Layer(train_way, sample_num)

    def set_forward(self, batch, ):
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_target.size(0)
        query_target = query_target.contiguous().reshape(episode_size, -1)

        class_feat = torch.mean(support_feat.contiguous().reshape(episode_size, self.train_way, self.train_shot, -1),
                                dim=2, keepdim=False)

        weight_mean = self.weight_mean(class_feat).permute((0, 2, 1))
        weight_logvar = self.weight_logvar(class_feat).permute((0, 2, 1))
        bias_mean = self.bias_mean(class_feat).permute((0, 2, 1))
        bias_logvar = self.bias_logvar(class_feat).permute((0, 2, 1))

        output, _ = self.head(query_feat, query_target, weight_mean, weight_logvar,
                                            bias_mean, bias_logvar)
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch, ):
        image, global_target= batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_target.size(0)
        query_target = query_target.contiguous().reshape(episode_size, -1)

        class_feat = torch.mean(support_feat.contiguous().reshape(episode_size, self.train_way, self.train_shot, -1),
                                dim=2, keepdim=False)

        weight_mean = self.weight_mean(class_feat).permute((0, 2, 1))
        weight_logvar = self.weight_logvar(class_feat).permute((0, 2, 1))
        bias_mean = self.bias_mean(class_feat).permute((0, 2, 1))
        bias_logvar = self.bias_logvar(class_feat).permute((0, 2, 1))

        output, task_score = self.head(query_feat, query_target, weight_mean, weight_logvar,
                                                     bias_mean, bias_logvar)
        acc = accuracy(output, query_target.reshape(-1))
        loss = -torch.mean(task_score, dim=0)
        return output, acc, loss

    def train_loop(self, *args, **kwargs):
        raise NotImplementedError

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError
