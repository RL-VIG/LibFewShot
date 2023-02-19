# -*- coding: utf-8 -*-
"""
@inproceedings{DeepBDC-CVPR2022,
    title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
    author={Jiangtao Xie and Fei Long and Jiaming Lv and Qilong Wang and Peihua Li}, 
    booktitle={CVPR},
    year={2022}
 }

Adapted from https://github.com/Fei-Long121/DeepBDC
"""

import torch
from torch import nn
import numpy as np
import copy
from core.utils import accuracy
import torch.nn.functional as F
from .finetuning_model import FinetuningModel
from ..metric.deepbdc import ProtoLayer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from .. import DistillKLLoss


# FIXME: Add multi-GPU support
class DistillLayer(nn.Module):
    def __init__(
        self,
        emb_func,
        cls_classifier,
        dropout_rate,
        is_distill,
        emb_func_path=None,
        cls_classifier_path=None,
    ):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
        self.cls_classifier = self._load_state_dict(
            cls_classifier, cls_classifier_path, is_distill
        )

        self.dropout = nn.Dropout(dropout_rate)


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
        if self.emb_func is not None and self.classifier is not None:
            output = self.emb_func(x)
            output = self.dropout(output)
            output = self.classifier(output)
        return output



class DeepBDC_Pretrain(FinetuningModel):
    def __init__(
        self, 
        num_class, 
        is_distill=False, 
        kd_T=4, 
        val_type='meta',
        reduce_dim=640,
        dropout_rate=0.5,
        penalty_C=0.1,
        emb_func_path=None,
        cls_classifier_path=None,
        **kwargs
    ):
        super(DeepBDC_Pretrain, self).__init__(**kwargs)
        self.num_class = num_class
        self.feat_dim = int(reduce_dim * (reduce_dim + 1) / 2)
        self.val_type = val_type
        # 1 shot 0.1, 5 shot 2
        self.penalty_C = penalty_C
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.classifier.bias.data.fill_(0)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.kl_loss_fn = DistillKLLoss(T=kd_T)    
        # DeepBDC's ProtoLayer
        self.meta_val_classifier = ProtoLayer()

        # distillation's part
        self.is_distill = is_distill
        self.distill_layer = DistillLayer(
            self.emb_func,
            self.classifier,
            dropout_rate,
            self.is_distill,
            emb_func_path,
            cls_classifier_path,
        )

    def set_forward(self, batch):
        if self.val_type == 'meta':
            output, acc = self.meta_set_forward(batch)
        elif self.val_type == 'stl':
            output, acc = self.stl_set_forward(batch)
        else:
            raise NotImplementedError("validation method are expected to be [meta, stl]")
        return output, acc

    def meta_set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
            feat = self.dropout(feat)
        
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        output = self.meta_val_classifier(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(-1, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def stl_set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        # support_feat -- [t, ws, c],       query_feat -- [t, wq, c], 
        # support_target -- [t, ws],        query_feat -- [t, wq]
        episode_size = support_feat.size(0)

        output_list = []
        acc_list = []
        for idx in range(episode_size):
            SF = support_feat[idx]      # [ws, c]
            QF = query_feat[idx]        # [ws, c]
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
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.dropout(feat)
        output = self.classifier(output)

        if self.is_distill:
            distill_output = self.distill_layer(image)
            loss = 0.5 * self.ce_loss_fn(output, target) \
                    + 0.5 * self.kl_loss_fn(output, distill_output)
        else:
            loss = self.ce_loss_fn(output, target)

        acc = accuracy(output, target)
        return output, acc, loss


    def set_forward_adaptation(self, support_feat, support_target):
        classifier = LogisticRegression(
            random_state=0,
            solver="lbfgs",
            C=self.penalty_C,
            max_iter=1000,
            penalty="l2",
            multi_class="multinomial",
        )
        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_target = support_target.detach().cpu().numpy()

        classifier.fit(support_feat, support_target)

        return classifier

    # def train(self, mode=True):
    #     super(DeepBDC_Pretrain, self).train(mode)
    #     self.distill_layer.train(mode)
