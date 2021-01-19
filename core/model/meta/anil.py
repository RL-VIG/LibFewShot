import copy

import torch
from torch import nn

from core.utils import accuracy
from .meta_model import MetaModel


class MLP(nn.Module):
    def __init__(self, feat_dim, hid_dim, way_num):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, way_num)
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
        images, targets = self.progress_batch2(batch)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))

        emb = self.model_func(images)
        emb = emb.contiguous().view(episode_size, self.way_num, self.shot_num + self.query_num, -1)
        targets = targets.contiguous().view(episode_size, self.way_num, self.shot_num + self.query_num)

        emb_support = emb[:, :, :self.shot_num, :].contiguous().view(episode_size, self.way_num * self.shot_num, -1)
        emb_query = emb[:, :, self.shot_num:, :].contiguous().view(episode_size, self.way_num * self.query_num, -1)
        support_targets = targets[:, :, :self.shot_num].contiguous().view(episode_size, -1)
        query_targets = targets[:, :, self.shot_num:].contiguous().view(-1)

        output_list = []
        for i in range(episode_size):
            classifier_copy = self.train_loop(emb_support[i], support_targets[i])
            output = classifier_copy(emb_query[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        prec1, _ = accuracy(output.squeeze(), query_targets, topk=(1, 3))
        return output, prec1

    def set_forward_loss(self, batch, ):
        images, targets = self.progress_batch2(batch)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))

        emb = self.model_func(images)
        emb = emb.contiguous().view(episode_size, self.way_num, self.shot_num + self.query_num, -1)
        targets = targets.contiguous().view(episode_size, self.way_num, self.shot_num + self.query_num)

        emb_support = emb[:, :, :self.shot_num, :].contiguous().view(episode_size, self.way_num * self.shot_num, -1)
        emb_query = emb[:, :, self.shot_num:, :].contiguous().view(episode_size, self.way_num * self.query_num, -1)
        support_targets = targets[:, :, :self.shot_num].contiguous().view(episode_size, -1)
        query_targets = targets[:, :, self.shot_num:].contiguous().view(-1)

        output_list = []
        for i in range(episode_size):
            classifier_copy = self.train_loop(emb_support[i], support_targets[i])
            output = classifier_copy(emb_query[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output.squeeze(), query_targets, topk=(1, 3))
        return output, prec1, loss

    def train_loop(self, support_feat, support_targets):
        support_targets = support_targets.detach()
        classifier = copy.deepcopy(self.classifier)
        optimizer = self.sub_optimizer(classifier.parameters(), self.inner_optim)

        classifier.train()
        for i in range(self.inner_train_iter):
            output = classifier(support_feat)

            loss = self.loss_func(output, support_targets)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        return classifier

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError
