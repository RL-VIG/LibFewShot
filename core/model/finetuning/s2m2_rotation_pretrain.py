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


import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .finetuning_model import FinetuningModel


class S2M2_Rotation_pretrain(FinetuningModel):
    def __init__(self, feat_dim, num_class, **kwargs):
        super(S2M2_Rotation_pretrain, self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.num_class = num_class

        self.cls_classifier = nn.Linear(self.feat_dim, self.num_class)
        self.rot_classifier = nn.Linear(self.num_class, 4)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        generated_image, generated_target, _ = self.rot_image_generation(image, target)

        feat = self.emb_func(generated_image)
        output = self.cls_classifier(feat)

        acc = accuracy(output, generated_target)

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        generated_image, generated_target, rot_target = self.rot_image_generation(image, target)

        feat = self.emb_func(generated_image)
        output = self.cls_classifier(feat)

        rot_output = self.rot_classifier(output)
        cls_loss = self.loss_func(output, generated_target)
        rot_loss = self.loss_func(rot_output, rot_target)

        loss = 0.5 * rot_loss + 0.5 * cls_loss

        acc = accuracy(output, generated_target)

        return output, acc, loss

    def rot_image_generation(self, image, target):
        batch_size = image.size(0)
        images_90 = image.transpose(2, 3).flip(2)
        images_180 = image.flip(2).flip(3)
        images_270 = image.flip(2).transpose(2, 3)

        generated_image = torch.cat([image, images_90, images_180, images_270], dim=0)
        generated_target = target.repeat(4)

        rot_target = torch.zeros(batch_size * 4)
        rot_target[batch_size:] += 1
        rot_target[batch_size * 2 :] += 1
        rot_target[batch_size * 3 :] += 1
        rot_target = rot_target.long().to(self.device)

        return generated_image, generated_target, rot_target
