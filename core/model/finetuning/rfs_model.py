# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/eccv/TianWKTI20,
  author    = {Yonglong Tian and
               Yue Wang and
               Dilip Krishnan and
               Joshua B. Tenenbaum and
               Phillip Isola}
  title     = {Rethinking Few-Shot Image Classification: {A} Good Embedding is All
               You Need?},
  booktitle = {Computer Vision - {ECCV} 2020 - 16th European Conference, Glasgow,
               UK, August 23-28, 2020, Proceedings, Part {XIV}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12359},
  pages     = {266--282},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-58568-6_16},
  doi       = {10.1007/978-3-030-58568-6_16}
}
https://arxiv.org/abs/2003.11539

Adapted from https://github.com/WangYueFt/rfs.
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
from .. import DistillKLLoss


# FIXME: Add multi-GPU support
class DistillLayer(nn.Module):
    def __init__(
        self,
        emb_func,
        classifier,
        is_distill,
        emb_func_path=None,
        classifier_path=None,
    ):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
        self.classifier = self._load_state_dict(classifier, classifier_path, is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distill):
        new_model = None
        if is_distill and state_dict_path is not None:
            new_model = copy.deepcopy(model)
            model_state_dict = torch.load(state_dict_path, map_location="cpu")
            new_model.load_state_dict(model_state_dict)
        return new_model

    @torch.no_grad()
    def forward(self, x):
        output = None
        if self.emb_func is not None and self.classifier is not None:
            output = self.emb_func(x)
            output = self.classifier(output)
        return output


class RFSModel(FinetuningModel):
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
        super(RFSModel, self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.num_class = num_class

        self.is_distill = is_distill
        self.gamma = gamma
        self.alpha = alpha

        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)

        self._init_network()

        self.distill_layer = DistillLayer(
            self.emb_func,
            self.classifier,
            self.is_distill,
            emb_func_path,
            classifier_path,
        )

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
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
