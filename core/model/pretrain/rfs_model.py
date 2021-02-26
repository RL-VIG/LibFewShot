import copy

import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .pretrain_model import PretrainModel
from .. import DistillKLLoss


class DistillLayer(nn.Module):
    def __init__(self, model_func, classifier, is_distill, model_func_path=None, classifier_path=None):
        super(DistillLayer, self).__init__()
        self.model_func = self._load_state_dict(model_func, model_func_path, is_distill)
        self.classifier = self._load_state_dict(classifier, classifier_path, is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distill):
        new_model = None
        if is_distill and state_dict_path is not None:
            new_model = copy.deepcopy(model)
            model_state_dict = torch.load(state_dict_path, map_location='cpu')
            new_model.load_state_dict(model_state_dict)
        return new_model

    @torch.no_grad()
    def forward(self, x):
        output = None
        if self.model_func is not None and self.classifier is not None:
            output = self.model_func(x)
            output = self.classifier(output)
        return output


class RFSModel(PretrainModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim,
                 num_classes, gamma=1, alpha=0, is_distill=False, kd_T=4,
                 model_func_path=None, classifier_path=None):
        super(RFSModel, self).__init__(way_num, shot_num, query_num, model_func, device)

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.is_distill = is_distill
        self.gamma = gamma
        self.alpha = alpha

        self.classifier = nn.Linear(self.feat_dim, self.num_classes)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)

        self._init_network()

        self.distill_layer = DistillLayer(self.model_func, self.classifier,
                                          self.is_distill, model_func_path, classifier_path)

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        images = images.to(self.device)
        with torch.no_grad():
            feat = self.model_func(images)
        support_feats, query_feats, support_targets, query_targets = self.split_by_episode(feat, mode=1)

        outputs = []
        prec1s = []
        for idx in range(episode_size):
            support_feat = support_feats[idx]
            query_feat = query_feats[idx]
            support_target = support_targets[idx]
            query_target = query_targets[idx]

            classifier = self.test_loop(support_feat, support_target)

            query_feat = F.normalize(query_feat, p=2, dim=1).detach().cpu().numpy()
            query_target = query_target.detach().cpu().numpy()

            output = classifier.predict(query_feat)
            prec1 = metrics.accuracy_score(query_target, output) * 100

            outputs.append(output)
            prec1s.append(prec1)

        output = np.stack(outputs, axis=0)
        prec1 = sum(prec1s) / episode_size
        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        feat = self.model_func(images)
        output = self.classifier(feat)
        distill_output = self.distill_layer(images)

        gamma_loss = self.ce_loss_func(output, targets)
        alpha_loss = self.kl_loss_func(output, distill_output)
        loss = gamma_loss * self.gamma + alpha_loss * self.alpha

        prec1, _ = accuracy(output, targets, topk=(1, 3))

        return output, prec1, loss

    def test_loop(self, support_feat, support_targets):
        return self.set_forward_adaptation(support_feat, support_targets)

    def set_forward_adaptation(self, support_feat, support_targets):
        classifier = LogisticRegression(penalty='l2',
                                        random_state=0,
                                        C=1.0,
                                        solver='lbfgs',
                                        max_iter=1000,
                                        multi_class='multinomial')

        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_targets = support_targets.detach().cpu().numpy()

        classifier.fit(support_feat, support_targets)

        return classifier

    def train(self, mode=True):
        self.model_func.train(mode)
        self.classifier.train(mode)
        self.distill_layer.train(False)

    def eval(self):
        super(RFSModel, self).eval()
