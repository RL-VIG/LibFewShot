# -*- coding: utf-8 -*-
"""
@article{DBLP:journals/corr/abs-2006-09785,
  author    = {Jathushan Rajasegaran and
               Salman Khan and
               Munawar Hayat and
               Fahad Shahbaz Khan and
               Mubarak Shah},
  title     = {Self-supervised Knowledge Distillation for Few-shot Learning},
  journal   = {CoRR},
  volume    = {abs/2006.09785},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.09785},
  archivePrefix = {arXiv},
  eprint    = {2006.09785}
}
https://arxiv.org/abs/2006.09785

Adapted from https://github.com/brjathu/SKD.
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
from core.model.loss import L2DistLoss


# FIXME: Add multi-GPU support
class DistillLayer(nn.Module):
    def __init__(
        self,
        emb_func,
        cls_classifier,
        is_distill,
        emb_func_path=None,
        cls_classifier_path=None,
    ):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
        self.cls_classifier = self._load_state_dict(
            cls_classifier, cls_classifier_path, is_distill
        )

    def _load_state_dict(self, model, state_dict_path, is_distill):
        new_model = None
        if is_distill and state_dict_path is not None:
            model_state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model_state_dict)
            new_model = copy.deepcopy(model)
        return new_model

    @torch.no_grad()
    def forward(self, x):
        output = None
        if self.emb_func is not None and self.cls_classifier is not None:
            output = self.emb_func(x)
            output = self.cls_classifier(output)

        return output


class SKDModel(FinetuningModel):
    def __init__(
        self,
        feat_dim,
        num_class,
        gamma=1,
        alpha=1,
        is_distill=False,
        kd_T=4,
        emb_func_path=None,
        cls_classifier_path=None,
        **kwargs
    ):
        super(SKDModel, self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.num_class = num_class

        self.gamma = gamma
        self.alpha = alpha

        self.is_distill = is_distill

        self.cls_classifier = nn.Linear(self.feat_dim, self.num_class)
        self.rot_classifier = nn.Linear(self.num_class, 4)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.l2_loss_func = L2DistLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)

        self.distill_layer = DistillLayer(
            self.emb_func,
            self.cls_classifier,
            self.is_distill,
            emb_func_path,
            cls_classifier_path,
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
            ST = support_target[idx].reshape(-1)
            QT = query_target[idx].reshape(-1)

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
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        batch_size = image.size(0)

        generated_image, generated_target, rot_target = self.rot_image_generation(
            image, target
        )

        feat = self.emb_func(generated_image)
        output = self.cls_classifier(feat)
        distill_output = self.distill_layer(image)

        if self.is_distill:
            gamma_loss = self.kl_loss_func(output[:batch_size], distill_output)
            alpha_loss = self.l2_loss_func(output[batch_size:], output[:batch_size]) / 3
        else:
            rot_output = self.rot_classifier(output)
            gamma_loss = self.ce_loss_func(output, generated_target)
            alpha_loss = torch.sum(
                F.binary_cross_entropy_with_logits(rot_output, rot_target)
            )

        loss = gamma_loss * self.gamma + alpha_loss * self.alpha

        acc = accuracy(output, generated_target)

        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = LogisticRegression(
            random_state=0,
            solver="lbfgs",
            max_iter=1000,
            penalty="l2",
            multi_class="multinomial",
        )

        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_target = support_target.detach().cpu().numpy()

        classifier.fit(support_feat, support_target)

        return classifier

    def rot_image_generation(self, image, target):
        batch_size = image.size(0)
        images_90 = image.transpose(2, 3).flip(2)
        images_180 = image.flip(2).flip(3)
        images_270 = image.flip(2).transpose(2, 3)

        if self.is_distill:
            generated_image = torch.cat((image, images_180), dim=0)
            generated_target = target.repeat(2)

            rot_target = torch.zeros(batch_size * 4)
            rot_target[batch_size:] += 1
            rot_target = rot_target.long().to(self.device)
        else:
            generated_image = torch.cat(
                [image, images_90, images_180, images_270], dim=0
            )
            generated_target = target.repeat(4)

            rot_target = torch.zeros(batch_size * 4)
            rot_target[batch_size:] += 1
            rot_target[batch_size * 2 :] += 1
            rot_target[batch_size * 3 :] += 1
            rot_target = (
                F.one_hot(rot_target.to(torch.int64), 4).float().to(self.device)
            )

        return generated_image, generated_target, rot_target
