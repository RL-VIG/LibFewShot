# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/LiWXHGL19,
  author    = {Wenbin Li and
               Lei Wang and
               Jinglin Xu and
               Jing Huo and
               Yang Gao and
               Jiebo Luo},
  title     = {Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot
               Learning},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}
               2019, Long Beach, CA, USA, June 16-20, 2019},
  pages     = {7260--7268},
  year      = {2019},
  url       = {http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Revisiting_Local_Descriptor_Based_Image-To
  -Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.html},
  doi       = {10.1109/CVPR.2019.00743}
}
https://arxiv.org/abs/1903.12290

Adapted from https://github.com/WenbinLee/DN4.
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class DN4Layer(nn.Module):
    def __init__(self, n_k):
        super(DN4Layer, self).__init__()
        self.n_k = n_k

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        query_num,
    ):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        query_feat = query_feat.view(t, way_num * query_num, c, h * w).permute(
            0, 1, 3, 2
        )
        query_feat = F.normalize(query_feat, p=2, dim=-1).unsqueeze(2)

        # t, ws, c, h, w -> t, w, s, c, hw -> t, 1, w, c, shw
        support_feat = (
            support_feat.view(t, way_num, shot_num, c, h * w)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(t, way_num, c, shot_num * h * w)
        )
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)

        # t, wq, w, hw, shw -> t, wq, w, hw, n_k -> t, wq, w
        relation = torch.matmul(query_feat, support_feat)
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        score = torch.sum(topk_value, dim=[3, 4])

        return score


class DN4(MetricModel):
    def __init__(self, n_k=3, **kwargs):
        super(DN4, self).__init__(**kwargs)
        self.dn4_layer = DN4Layer(n_k)
        self.loss_func = nn.CrossEntropyLoss()

        if self.jigsaw:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier = nn.Sequential()
            self.classifier.add_module('fc8',nn.Linear(4096, 35))

        if self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(512,128))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )

        output = self.dn4_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )

        output = self.dn4_layer(
            query_feat,
            support_feat,
            self.way_num,
            self.shot_num,
            self.query_num,
        ).view(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss

    def set_forward_SS(self, patches=None, patches_label=None):
        if len(patches.size()) == 6:
            Way,S,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
            B = Way*S
        elif len(patches.size()) == 5:
            B,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
        if self.jigsaw:
            patches = patches.view(B*T,C,H,W).cuda()#torch.Size([675, 3, 64, 64])
            # if self.dual_cbam:
            #     patch_feat = self.feature(patches, jigsaw=True)#torch.Size([675, 512])
            # else:
            #     patch_feat = self.feature(patches)#torch.Size([675, 512])
            patch_feat = self.emb_func(patches)#torch.Size([675, 512])
            if not self.emb_func.avg_pool:
                patch_feat = nn.AdaptiveAvgPool2d((1, 1))(patch_feat)

            if not self.emb_func.is_flatten:
                patch_feat = patch_feat.view(patch_feat.size(0), -1)
            
            x_ = patch_feat.view(B,T,-1)
            x_ = x_.transpose(0,1)#torch.Size([9, 75, 512])

            x_list = []
            for i in range(9):
                z = self.fc6(x_[i])#torch.Size([75, 512])
                z = z.view([B,1,-1])#torch.Size([75, 1, 512])
                x_list.append(z)

            x_ = torch.cat(x_list,1)#torch.Size([75, 9, 512])
            x_ = self.fc7(x_.view(B,-1))#torch.Size([75, 9*512])
            x_ = self.classifier(x_)

            y_ = patches_label.view(-1).cuda()

            return x_, y_
        elif self.rotation:
            patches = patches.view(B*T,C,H,W).cuda()
            x_ = self.emb_func(patches)#torch.Size([64, 512, 1, 1])
            if not self.emb_func.avg_pool:
                x_ = nn.AdaptiveAvgPool2d((1, 1))(x_)

            if not self.emb_func.is_flatten:
                x_ = x_.view(x_.size(0), -1)

            x_ = x_.squeeze()
            x_ = self.fc6(x_)
            x_ = self.fc7(x_)#64,128
            x_ = self.classifier_rotation(x_)#64,4
            pred = torch.max(x_,1)
            y_ = patches_label.view(-1).cuda()
            return x_, y_

    def set_forward_loss_SS(self, patches=None, patches_label=None):
        if self.jigsaw:
            x_, y_ = self.set_forward_SS(patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
        elif self.rotation:
            x_, y_ = self.set_forward_SS(patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_rotation = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)

        if self.jigsaw:
            return self.loss_func(x_,y_), acc_jigsaw
        elif self.rotation:
            return self.loss_func(x_,y_), acc_rotation
