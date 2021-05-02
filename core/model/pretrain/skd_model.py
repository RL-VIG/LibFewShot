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


class L2DistLoss(nn.Module):
    def __init__(self):
        super(L2DistLoss, self).__init__()

    def forward(self, feat1, feat2):
        loss = torch.mean(torch.sqrt(torch.sum((feat1 - feat2) ** 2, dim=1)))
        if torch.isnan(loss).any():
            loss = 0.0
        return loss


class DistillLayer(nn.Module):
    def __init__(self, emb_func, cls_classifier, is_distill,
                 emb_func_path=None, cls_classifier_path=None, ):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
        self.cls_classifier = self._load_state_dict(cls_classifier, cls_classifier_path,
                                                    is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distill):
        new_model = None
        if is_distill and state_dict_path is not None:
            model_state_dict = torch.load(state_dict_path, map_location='cpu')
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


class SKDModel(PretrainModel):
    def __init__(self, way_num, shot_num, query_num, emb_func, device, feat_dim,
                 num_class, gamma=1, alpha=1, is_distill=False, kd_T=4,
                 emb_func_path=None, cls_classifier_path=None, ):
        super(SKDModel, self).__init__(way_num, shot_num, query_num, emb_func, device, )

        self.feat_dim = feat_dim
        self.num_class = num_class

        self.gamma = gamma
        self.alpha = alpha

        self.is_distill = is_distill

        self.cls_classifier = nn.Linear(self.feat_dim, self.num_class)
        self.rot_classifier = nn.Linear(self.num_classes, 4)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.l2_loss_func = L2DistLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)

        self._init_network()

        self.distill_layer = DistillLayer(self.emb_func, self.cls_classifier,
                                          self.is_distill, emb_func_path,
                                          cls_classifier_path, )

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target \
            = self.split_by_episode(feat, mode=1)

        output_list = []
        prec1_list = []
        for idx in range(episode_size):
            support_feat = support_feat[idx]
            query_feat = query_feat[idx]
            support_target = support_target[idx]
            query_target = query_target[idx]

            classifier = self.test_loop(support_feat, support_target)

            query_feat = F.normalize(query_feat, p=2, dim=1).detach().cpu().numpy()
            query_target = query_target.detach().cpu().numpy()

            output = classifier.predict(query_feat)
            prec1 = metrics.accuracy_score(query_target, output) * 100

            output_list.append(output)
            prec1_list.append(prec1)

        output = np.stack(output_list, axis=0)
        prec1 = sum(prec1_list) / episode_size
        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        batch_size = image.size(0)

        generated_image, generated_target, rot_target \
            = self.rot_image_generation(image, target)

        feat = self.emb_func(generated_image)
        cls_output = self.cls_classifier(feat)
        distill_output = self.distill_layer(image)

        if self.is_distill:
            gamma_loss = self.kl_loss_func(cls_output[:batch_size], distill_output)
            alpha_loss = self.l2_loss_func(cls_output[batch_size:],
                                           cls_output[:batch_size]) / 3
        else:
            rot_output = self.rot_classifier(cls_output)
            gamma_loss = self.ce_loss_func(cls_output, generated_target)
            alpha_loss = torch.sum(
                F.binary_cross_entropy_with_logits(rot_output, rot_target))

        loss = gamma_loss * self.gamma + alpha_loss * self.alpha

        prec1, _ = accuracy(cls_output, generated_target, topk=(1, 3))

        return cls_output, prec1, loss

    def test_loop(self, support_feat, support_target):
        return self.set_forward_adaptation(support_feat, support_target)

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = LogisticRegression(random_state=0,
                                        solver='lbfgs',
                                        max_iter=1000,
                                        penalty='l2',
                                        multi_class='multinomial')

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
            generated_image = torch.cat((image, images_90, images_180, images_270),
                                         dim=0)
            generated_target = target.repeat(4)

            rot_target = torch.zeros(batch_size * 4)
            rot_target[batch_size:] += 1
            rot_target[batch_size * 2:] += 1
            rot_target[batch_size * 3:] += 1
            rot_target = F.one_hot(rot_target.to(torch.int64), 4) \
                .float().to(self.device)

        return generated_image, generated_target, rot_target

    def train(self, mode=True):
        self.emb_func.train(mode)
        self.rot_classifier.train(mode)
        self.cls_classifier.train(mode)
        self.distill_layer.train(False)

    def eval(self):
        super(SKDModel, self).eval()
