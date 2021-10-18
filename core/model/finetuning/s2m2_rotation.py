# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/wacv/Mangla0SKBK20,
  author    = {Puneet Mangla and
               Mayank Singh and
               Abhishek Sinha and
               Nupur Kumari and
               Vineeth N. Balasubramanian and
               Balaji Krishnamurthy},
  title     = {Charting the Right Manifold: Manifold Mixup for Few-shot Learning},
  booktitle = {{IEEE} Winter Conference on Applications of Computer Vision, {WACV}
               2020, Snowmass Village, CO, USA, March 1-5, 2020},
  pages     = {2207--2216},
  year      = {2020},
  url       = {https://doi.org/10.1109/WACV45572.2020.9093338},
  doi       = {10.1109/WACV45572.2020.9093338},
}
http://arxiv.org/abs/1907.12087

Adapted from https://github.com/nupurkmr9/S2M2_fewshot.
"""
import copy

import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .finetuning_model import FinetuningModel


class S2M2_Rotation_Model(FinetuningModel):
    def __init__(
        self,
        feat_dim,
        num_class,
        gamma=1,
        alpha=0,
        is_distill=False,
        kd_T=4,
        emb_func_path=None,
        classifier_path=None,
        **kwargs
    ):
        super(S2M2_Rotation_Model, self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.num_class = num_class

        self.is_distill = is_distill
        self.gamma = gamma
        self.alpha = alpha

        self.rotate_classifier = nn.Linear(640, 4)

        self._init_network()

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        output_list = []
        acc_list = []
        for idx in range(episode_size):
            SF = support_feat[idx]
            QF = query_feat[idx]
            ST = support_target[idx]
            QT = query_target[idx]

            classifier = self.set_forward_adaptation(SF, ST)

            QF = F.normalize(QF, p=2, dim=1).detach().cpu().numpy()
            QT = QT.detach().cpu().numpy()

            output = classifier.predict(QF)
            acc = metrics.accuracy_score(QT, output) * 100

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

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = LogisticRegression(
            penalty="l2",
            random_state=0,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
        )

        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_target = support_target.detach().cpu().numpy()

        classifier.fit(support_feat, support_target)

        return classifier

    def train(self, mode=True):
        self.emb_func.train(mode)
        self.classifier.train(mode)
        self.distill_layer.train(False)

    def eval(self):
        super(S2M2_Rotation_Model, self).eval()
