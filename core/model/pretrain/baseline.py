import torch
from torch import nn

from core.utils import accuracy
from .pretrain_model import PretrainModel


class Baseline(PretrainModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim,
                 num_classes, inner_optim=None, inner_train_iter=20):
        super(Baseline, self).__init__(way_num, shot_num, query_num, model_func, device)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.inner_optim = inner_optim
        self.inner_train_iter = inner_train_iter

        self.classifier = nn.Linear(self.feat_dim, self.num_classes)
        self.loss_func = nn.CrossEntropyLoss()

        self._init_network()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
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
            query_feat = self.model_func(query_images)

        classifier = self.test_loop(support_feat, support_targets)

        output = classifier(query_feat)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, targets = batch
        images = torch.cat(images, 0)
        targets = torch.cat(targets, 0)
        images = images.to(self.device)
        targets = targets.to(self.device)

        feat = self.model_func(images)
        output = self.classifier(feat)
        loss = self.loss_func(output, targets)
        prec1, _ = accuracy(output, targets, topk=(1, 3))
        return output, prec1, loss

    def test_loop(self, support_feat, support_targets):
        return self.set_forward_adaptation(support_feat, support_targets)

    def set_forward_adaptation(self, support_feat, support_targets):
        classifier = nn.Linear(self.feat_dim, self.way_num)
        optimizer = self.sub_optimizer(classifier, self.inner_optim)

        classifier = classifier.to(self.device)

        classifier.train()
        for i in range(self.inner_train_iter):
            output = classifier(support_feat)

            loss = self.loss_func(output, support_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return classifier
