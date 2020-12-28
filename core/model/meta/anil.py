import torch
from torch import nn
import copy

from core.model.abstract_model import AbstractModel
from core.utils import accuracy


class Classifier(nn.Module):
    def __init__(self, feat_dim=64, hid_dim=128, way_num=5) -> None:
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, way_num)
        )

    def forward(self, x):
        return self.layers(x)


class ANIL(AbstractModel):
    def __init__(self, way_num, shot_num, query_num, feature, device, feat_dim=1600, hid_dim=800, inner_optim=None,
                 inner_train_iter=10):
        super(ANIL, self).__init__(way_num, shot_num, query_num, feature, device)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = Classifier(feat_dim, hid_dim, way_num=way_num)
        self.inner_optim = inner_optim
        self.inner_train_iter = inner_train_iter
        self._init_network()

    def set_forward(self, batch, ):
        query_images, query_targets, support_images, support_targets = batch
        query_images = torch.cat(query_images, 0)
        query_targets = torch.cat(query_targets, 0)
        support_images = torch.cat(support_images, 0)
        support_targets = torch.cat(support_targets, 0)
        query_images = query_images.to(self.device)
        query_targets = query_targets.to(self.device)
        support_images = support_images.to(self.device)
        support_targets = support_targets.to(self.device)

        with torch.no_grad():
            support_feat = self.model_func(support_images)
        classifier_copy = self.train_loop(support_feat, support_targets)

        query_feat = self.model_func(query_images)
        output = classifier_copy(query_feat)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1

    def set_forward_loss(self, batch, ):
        query_images, query_targets, support_images, support_targets = batch
        query_images = torch.cat(query_images, 0)
        query_targets = torch.cat(query_targets, 0)
        support_images = torch.cat(support_images, 0)
        support_targets = torch.cat(support_targets, 0)
        query_images = query_images.to(self.device)
        query_targets = query_targets.to(self.device)
        support_images = support_images.to(self.device)
        support_targets = support_targets.to(self.device)

        with torch.no_grad():
            support_feat = self.model_func(support_images)
        classifier_copy = self.train_loop(support_feat, support_targets)

        query_feat = self.model_func(query_images)
        output = classifier_copy(query_feat)
        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1, loss

    def train_loop(self, support_feat, support_targets):
        classifier = copy.deepcopy(self.classifier)
        optimizer = self.sub_optimizer(classifier, self.inner_optim)

        classifier.train()
        for i in range(self.inner_train_iter):
            output = classifier(support_feat)

            loss = self.loss_func(output, support_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return classifier

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError
