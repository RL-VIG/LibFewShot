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

# FIXME 加上多GPU

#https://github.com/WangYueFt/rfs
class DistillLayer(nn.Module):
    def __init__(self, emb_func, classifier, is_distill, emb_func_path=None, classifier_path=None):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
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
        if self.emb_func is not None and self.classifier is not None:
            output = self.emb_func(x)
            output = self.classifier(output)
        return output


class RFSModel(PretrainModel):
    def __init__(self, train_way, train_shot, train_query, emb_func, device, feat_dim,
                 num_class, gamma=1, alpha=0, is_distill=False, kd_T=4,
                 emb_func_path=None, classifier_path=None):
        super(RFSModel, self).__init__(train_way, train_shot, train_query, emb_func, device)

        self.feat_dim = feat_dim
        self.num_class = num_class

        self.is_distill = is_distill
        self.gamma = gamma
        self.alpha = alpha

        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)

        self._init_network()

        self.distill_layer = DistillLayer(self.emb_func, self.classifier,
                                          self.is_distill, emb_func_path, classifier_path)

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        episode_size = image.size(0) // (self.train_way * (self.train_shot + self.train_query))
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)

        output_list = []
        acc_list = []
        for idx in range(episode_size):
            support_feat = support_feat[idx]
            query_feat = query_feat[idx]
            support_target = support_target[idx]
            query_target = query_target[idx]

            classifier = self.test_loop(support_feat, support_target)

            query_feat = F.normalize(query_feat, p=2, dim=1).detach().cpu().numpy()
            query_target = query_target.detach().cpu().numpy()

            output = classifier.predict(query_feat)
            acc = metrics.accuracy_score(query_target, output) * 100

            output_list.append(output)
            acc_list.append(acc)

        output = np.stack(output_list, axis=0)
        acc = sum(acc_list) / episode_size
        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)
        distill_output = self.distill_layer(image)

        gamma_loss = self.ce_loss_func(output, global_target)
        alpha_loss = self.kl_loss_func(output, distill_output)
        loss = gamma_loss * self.gamma + alpha_loss * self.alpha

        acc = accuracy(output, global_target)

        return output, acc, loss

    def test_loop(self, support_feat, support_target):
        return self.set_forward_adaptation(support_feat, support_target)

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = LogisticRegression(penalty='l2',
                                        random_state=0,
                                        C=1.0,
                                        solver='lbfgs',
                                        max_iter=1000,
                                        multi_class='multinomial')

        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_target = support_target.detach().cpu().numpy()

        classifier.fit(support_feat, support_target)

        return classifier

    def train(self, mode=True):
        self.emb_func.train(mode)
        self.classifier.train(mode)
        self.distill_layer.train(False)

    def eval(self):
        super(RFSModel, self).eval()
