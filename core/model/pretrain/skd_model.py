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
from core.model.loss import L2DistLoss


class DistillLayer(nn.Module):
    def __init__(self, model_func, cls_classifier, is_distill,
                 model_func_path=None, cls_classifier_path=None, ):
        super(DistillLayer, self).__init__()
        self.model_func = self._load_state_dict(model_func, model_func_path, is_distill)
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
        if self.model_func is not None and self.cls_classifier is not None:
            output = self.model_func(x)
            output = self.cls_classifier(output)

        return output


class SKDModel(PretrainModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim,
                 num_classes, gamma=1, alpha=1, is_distill=False, kd_T=4,
                 model_func_path=None, cls_classifier_path=None, ):
        super(SKDModel, self).__init__(way_num, shot_num, query_num, model_func, device, )

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.gamma = gamma
        self.alpha = alpha

        self.is_distill = is_distill

        self.cls_classifier = nn.Linear(self.feat_dim, self.num_classes)
        self.rot_classifier = nn.Linear(self.num_classes, 4)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.l2_loss_func = L2DistLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)


        self.distill_layer = DistillLayer(self.model_func, self.cls_classifier,
                                          self.is_distill, model_func_path,
                                          cls_classifier_path, )

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
        support_feats, query_feats, support_targets, query_targets \
            = self.split_by_episode(feat, mode=1)

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

        batch_size = images.size(0)

        generated_images, generated_targets, rot_targets \
            = self.rot_image_generation(images, targets)

        feat = self.model_func(generated_images)
        cls_output = self.cls_classifier(feat)
        distill_output = self.distill_layer(images)

        if self.is_distill:
            gamma_loss = self.kl_loss_func(cls_output[:batch_size], distill_output)
            alpha_loss = self.l2_loss_func(cls_output[batch_size:],
                                           cls_output[:batch_size]) / 3
        else:
            rot_output = self.rot_classifier(cls_output)
            gamma_loss = self.ce_loss_func(cls_output, generated_targets)
            alpha_loss = torch.sum(
                F.binary_cross_entropy_with_logits(rot_output, rot_targets))

        loss = gamma_loss * self.gamma + alpha_loss * self.alpha

        prec1, _ = accuracy(cls_output, generated_targets, topk=(1, 3))

        return cls_output, prec1, loss

    def test_loop(self, support_feat, support_targets):
        return self.set_forward_adaptation(support_feat, support_targets)

    def set_forward_adaptation(self, support_feat, support_targets):
        classifier = LogisticRegression(random_state=0,
                                        solver='lbfgs',
                                        max_iter=1000,
                                        penalty='l2',
                                        multi_class='multinomial')

        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_targets = support_targets.detach().cpu().numpy()

        classifier.fit(support_feat, support_targets)

        return classifier

    def rot_image_generation(self, images, targets):
        batch_size = images.size(0)
        images_90 = images.transpose(2, 3).flip(2)
        images_180 = images.flip(2).flip(3)
        images_270 = images.flip(2).transpose(2, 3)

        if self.is_distill:
            generated_images = torch.cat((images, images_180), dim=0)
            generated_targets = targets.repeat(2)

            rot_targets = torch.zeros(batch_size * 4)
            rot_targets[batch_size:] += 1
            rot_targets = rot_targets.long().to(self.device)
        else:
            generated_images = torch.cat((images, images_90, images_180, images_270),
                                         dim=0)
            generated_targets = targets.repeat(4)

            rot_targets = torch.zeros(batch_size * 4)
            rot_targets[batch_size:] += 1
            rot_targets[batch_size * 2:] += 1
            rot_targets[batch_size * 3:] += 1
            rot_targets = F.one_hot(rot_targets.to(torch.int64), 4) \
                .float().to(self.device)

        return generated_images, generated_targets, rot_targets

    def train(self, mode=True):
        self.model_func.train(mode)
        self.rot_classifier.train(mode)
        self.cls_classifier.train(mode)
        self.distill_layer.train(False)

    def eval(self):
        super(SKDModel, self).eval()
