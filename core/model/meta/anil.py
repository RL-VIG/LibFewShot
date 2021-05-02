import copy

import torch
from torch import nn
import torch.nn.functional as F

from core.utils import accuracy
from .meta_model import MetaModel

class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class MLP(nn.Module):
    def __init__(self, feat_dim, hid_dim, way_num):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(feat_dim, hid_dim)
            Linear_fw(feat_dim, way_num)
        )

    def forward(self, x):
        return self.layers(x)


class Classifier(nn.Module):
    def __init__(self, way_num, feat_dim=64, hid_dim=128, inner_train_iter=10):
        super(Classifier, self).__init__()
        self.way_num = way_num
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.inner_train_iter = inner_train_iter

    def set_optimizer_func(self, optimizer_func, inner_optim):
        self.optimizer_func = optimizer_func
        self.inner_optim = inner_optim

    def set_loss(self, loss_func):
        self.loss_func = loss_func

    def forward(self, query_feat, support_feat, support_targets):
        assert (query_feat.size(0) == support_feat.size(0))
        episode_size = query_feat.size(0)
        output_list = []
        for episode in range(episode_size):
            net = MLP(self.feat_dim, self.hid_dim, self.way_num).to(query_feat.device)
            optimizer = self.optimizer_func(net.parameters(), self.inner_optim)
            net.train()
            episode_support_feat = support_feat[episode, :].detach()
            episode_query_feat = query_feat[episode, :]
            episode_support_targets = support_targets[episode, :]
            for i in range(self.inner_train_iter):
                output = net(episode_support_feat)

                loss = self.loss_func(output, episode_support_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            output_list.append(net(episode_query_feat))
        output = torch.cat(output_list, dim=0)
        return output


class ANIL(MetaModel):
    def __init__(self, way_num, shot_num, query_num, feature, device, feat_dim=1600,
                 hid_dim=800, inner_optim=None,
                 inner_train_iter=10):
        super(ANIL, self).__init__(way_num, shot_num, query_num, feature, device)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        # self.classifier = Classifier(way_num, feat_dim=feat_dim, hid_dim=hid_dim,
        #                              inner_train_iter=inner_train_iter)
        self.classifier = MLP(feat_dim=feat_dim, hid_dim=hid_dim, way_num=way_num)
        self.inner_optim = inner_optim
        self.inner_train_iter = inner_train_iter

        # self.classifier.set_optimizer_func(self.sub_optimizer, inner_optim)
        # self.classifier.set_loss(self.loss_func)
        self._init_network()

    def set_forward(self, batch, ):
        images, global_targets = batch
        images = images.to(self.device)

        emb = self.emb_func(images)
        emb_support, emb_query, support_targets, query_targets = self.split_by_episode(emb, mode=1)
        episode_size = emb_support.size(0)

        output_list = []
        for i in range(episode_size):
            self.train_loop(emb_support[i], support_targets[i])
            output = self.classifier(emb_query[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        prec1, _ = accuracy(output.squeeze(), query_targets.contiguous().reshape(-1), topk=(1, 3))
        return output, prec1

    def set_forward_loss(self, batch, ):
        images, global_targets = batch
        images = images.to(self.device)

        emb = self.emb_func(images)
        emb_support, emb_query, support_targets, query_targets = self.split_by_episode(emb, mode=1)
        episode_size = emb_support.size(0)

        output_list = []
        for i in range(episode_size):
            self.train_loop(emb_support[i], support_targets[i])
            output = self.classifier(emb_query[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_targets.contiguous().reshape(-1))
        prec1, _ = accuracy(output.squeeze(), query_targets.contiguous().reshape(-1), topk=(1, 3))
        return output, prec1, loss

    def train_loop(self, support_feat, support_targets):
        lr = self.inner_optim['lr']
        fast_parameters = list(self.classifier.parameters())
        for parameter in self.classifier.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()

        for i in range(self.inner_train_iter):
            output = self.classifier(support_feat)
            loss = self.loss_func(output, support_targets)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.classifier.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - self.inner_optim['lr'] * grad[k]
                fast_parameters.append(weight.fast)

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError
