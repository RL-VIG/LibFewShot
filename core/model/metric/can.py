"""
following: https://github.com/blue-blue272/fewshot-CAN/blob/master/torchFewShot/models/cam.py
"""
from __future__ import absolute_import
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from core.model.metric.metric_model import MetricModel


def one_hot(indice, depth, use_cuda=True):
    if use_cuda:
        encoded_indicie = torch.zeros(indice.size() + torch.Size([depth])).cuda()
    else:
        encoded_indicie = torch.zeros(indice.size() + torch.Size([depth]))
    index = indice.view(indice.size() + torch.Size([1]))
    encoded_indicie = encoded_indicie.scatter_(1, index, 1)
    return encoded_indicie


def shuffle(image, target, global_target):
    """
    A trick for CAN training
    """
    batch_size, sample_num = image.shape[0], image.shape[1]
    for i in range(4):
        indice = torch.randperm(sample_num).to(image.device)
        image[i] = image[i][indice]
        target[i] = target[i][indice]
        global_target[i] = global_target[i][indice]
    return image, target, global_target


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.
    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    """
    Support & Query share one attention
    """

    def __init__(self, mid_channels):
        super(CAM, self).__init__()
        # self.conv1 = ConvBlock(25, 5, 1)
        # self.conv2 = nn.Conv2d(5, 25, 1, stride=1, padding=0)
        self.conv1 = ConvBlock(mid_channels * mid_channels, mid_channels, 1)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels * mid_channels, 1, stride=1, padding=0
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def get_attention(self, a):
        input_a = a

        a = a.mean(3)  # GAP
        a = a.transpose(1, 3)
        a = F.relu(self.conv1(a))
        a = self.conv2(a)
        a = a.transpose(1, 3)
        a = a.unsqueeze(3)

        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a

    def forward(self, f1, f2):
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)

        # Flatten
        f1 = f1.view(b, n1, c, -1)
        f2 = f2.view(b, n2, c, -1)

        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)  # [1, 5, 512, 36]
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)  # [1, 75, 512, 36]
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
        f2_norm = f2_norm.unsqueeze(1)

        a1 = torch.matmul(f1_norm, f2_norm)  # [1, 5, 75, 36, 36]
        a2 = a1.transpose(3, 4)  # [1, 5, 75, 36, 36]

        a1 = self.get_attention(a1)  # [1, 5, 75, 36]
        a2 = self.get_attention(a2)  # [1, 5, 75, 36]

        f1 = f1.unsqueeze(2) * a1.unsqueeze(3)
        f1 = f1.view(b, n1, n2, c, h, w)
        f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
        f2 = f2.view(b, n1, n2, c, h, w)
        return f1.transpose(1, 2), f2.transpose(1, 2)


class CAMLayer(nn.Module):
    def __init__(
        self, scale_cls, iter_num_prob=35.0 / 75, num_classes=64, feat_dim=512, HW=5
    ):
        super(CAMLayer, self).__init__()
        self.scale_cls = scale_cls
        self.cam = CAM(HW)
        self.iter_num_prob = iter_num_prob
        self.feat_dim = feat_dim
        self.classifier = nn.Conv2d(self.feat_dim, num_classes, kernel_size=1)

    def val(self, support_feat, query_feat):
        query_feat = query_feat.mean(4)
        query_feat = query_feat.mean(4)
        support_feat = F.normalize(
            support_feat, p=2, dim=support_feat.dim() - 1, eps=1e-12
        )  # [1, 75, 5, 512]
        query_feat = F.normalize(
            query_feat, p=2, dim=query_feat.dim() - 1, eps=1e-12
        )  # [1, 75, 5, 512]
        scores = self.scale_cls * torch.sum(
            query_feat * support_feat, dim=-1
        )  # [1, 75, 5]
        return scores

    def forward(self, support_feat, query_feat, support_target, query_target):
        """
        support_feat: [4, 5, 512, 6, 6]
        query_feat: [4, 75, 512, 6, 6]
        support_target: [4, 5, 5] one-hot
        query_target: [4, 75, 5] one-hot
        """
        original_feat_shape = support_feat.size()
        batch_size = support_feat.size(0)
        n_support = support_feat.size(1)
        n_query = query_feat.size(1)
        way_num = support_target.size(-1)
        # feat = support_feat.size(-1)

        # flatten feature
        support_feat = support_feat.view(batch_size, n_support, -1)

        labels_train_transposed = support_target.transpose(1, 2)  # [1, 5, 5]

        # calc the prototypes of support set
        prototype = torch.bmm(
            labels_train_transposed, support_feat
        )  # [1, 5, 5]x[1, 5, 640]
        prototype = prototype.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototype)
        )  # [1, 5, 640]
        prototype = prototype.reshape(
            batch_size, -1, *original_feat_shape[2:]
        )  # [1, 5, 512, 6, 6]
        prototype, query_feat = self.cam(
            prototype, query_feat
        )  # [1, 75, 5, 512, 6, 6]  # [2, 75, 640, 1, 1]
        prototype = prototype.mean(4)
        prototype = prototype.mean(4)  # [1, 75, 5, 512]

        if not self.training:
            return self.val(prototype, query_feat)

        proto_norm = F.normalize(prototype, p=2, dim=3, eps=1e-12)
        query_norm = F.normalize(query_feat, p=2, dim=3, eps=1e-12)
        proto_norm = proto_norm.unsqueeze(4)
        proto_norm = proto_norm.unsqueeze(5)
        cls_score = self.scale_cls * torch.sum(query_norm * proto_norm, dim=3)
        cls_score = cls_score.view(batch_size * n_query, *cls_score.size()[2:])

        query_feat = query_feat.view(batch_size, n_query, way_num, -1)
        query_feat = query_feat.transpose(2, 3)
        query_target = query_target.unsqueeze(3)
        query_feat = torch.matmul(query_feat, query_target)
        query_feat = query_feat.view(
            batch_size * n_query, -1, *original_feat_shape[-2:]
        )
        query_target = self.classifier(query_feat)

        return query_target, cls_score

    def helper(self, support_feat, query_feat, support_target):
        """
        support_target_transposed: one-hot
        """
        b, n, c, h, w = support_feat.size()
        k = support_target.size(2)

        support_target_transposed = support_target.transpose(1, 2)
        support_feat = torch.bmm(support_target_transposed, support_feat.view(b, n, -1))
        support_feat = support_feat.div(
            support_target_transposed.sum(dim=2, keepdim=True).expand_as(support_feat)
        )
        support_feat = support_feat.view(b, -1, c, h, w)

        support_feat, query_feat = self.cam(support_feat, query_feat)
        support_feat = support_feat.mean(-1).mean(-1)
        query_feat = query_feat.mean(-1).mean(-1)

        query_feat = F.normalize(query_feat, p=2, dim=query_feat.dim() - 1, eps=1e-12)
        support_feat = F.normalize(
            support_feat, p=2, dim=support_feat.dim() - 1, eps=1e-12
        )
        score = self.scale_cls * torch.sum(query_feat * support_feat, dim=-1)
        return score

    def val_transductive(self, support_feat, query_feat, support_target, query_target):
        iter_num_prob = self.iter_num_prob
        batch_size, num_train = support_feat.size(0), support_feat.size(1)
        num_test = query_feat.size(1)
        K = support_target.size(2)

        cls_score = self.helper(support_feat, query_feat, support_target)

        num_image_per_iter = int(num_test * iter_num_prob)
        num_iter = num_test // num_image_per_iter

        for i in range(num_iter):
            max_score, pred = torch.max(cls_score, 2)
            chose_index = torch.argsort(max_score.view(-1), descending=True)
            chose_index = chose_index[: num_image_per_iter * (i + 1)]
            ftest_iter = query_feat[0, chose_index].unsqueeze(0)
            pred_iter = pred[0, chose_index].unsqueeze(0)

            pred_iter = one_hot(pred_iter.view(-1), K).cuda()
            pred_iter = pred_iter.view(batch_size, -1, K)

            support_feat_iter = torch.cat((support_feat, ftest_iter), 1)
            support_target_iter = torch.cat((support_target, pred_iter), 1)
            cls_score = self.helper(support_feat_iter, query_feat, support_target_iter)

        return cls_score


class CAN(MetricModel):
    def __init__(
        self,
        way_num,
        shot_num,
        query_num,
        emb_func,
        device,
        scale_cls,
        iter_num_prob=35.0 / 75,
        num_classes=64,
        feat_dim=512,
        HW=5,
    ):
        super(CAN, self).__init__(way_num, shot_num, query_num, emb_func, device)
        self.camLayer = CAMLayer(scale_cls, iter_num_prob, num_classes, feat_dim, HW)
        self.loss_func = nn.CrossEntropyLoss()

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

        # convert to one-hot
        support_target_one_hot = one_hot(
            support_target.view(episode_size * self.way_num * self.shot_num),
            self.way_num,
        )
        support_target_one_hot = support_target_one_hot.view(
            episode_size, self.way_num * self.shot_num, self.way_num
        )
        query_target_one_hot = one_hot(
            query_target.view(episode_size * self.way_num * self.query_num),
            self.way_num,
        )
        query_target_one_hot = query_target_one_hot.view(
            episode_size, self.way_num * self.query_num, self.way_num
        )
        output = self.cam_layer(
            support_feat, query_feat, support_target_one_hot, query_target_one_hot
        )
        # output = self.cam_layer.val_transductive(support_feat, query_feat, support_target_one_hot, query_target_one_hot)

        output = output.view(episode_size * self.way_num * self.query_num, -1)
        acc = accuracy(output, query_target)
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
        feat = self.emb_func(image)  # [80, 640]
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )  # [4,5,512,6,6] [4, 75, 512,6,6] [4, 5] [300]
        support_global_targets, query_global_targets = (
            global_target[:, :, : self.shot_num],
            global_target[:, :, self.shot_num :],
        )
        # # TODO: Shuffle label index
        # support_feat, support_target, support_global_targets = shuffle(support_feat, support_target, support_global_targets.reshape(*support_target.size()))
        # query_feat, query_target, query_global_targets = shuffle(query_feat, query_target.reshape(*query_feat.size()[:2]), query_global_targets.reshape(*query_feat.size()[:2]))

        # convert to one-hot
        support_target_one_hot = one_hot(
            support_target.view(episode_size * self.way_num * self.shot_num),
            self.way_num,
        )
        support_target_one_hot = support_target_one_hot.view(
            episode_size, self.way_num * self.shot_num, self.way_num
        )
        query_target_one_hot = one_hot(
            query_target.view(episode_size * self.way_num * self.query_num),
            self.way_num,
        )
        query_target_one_hot = query_target_one_hot.view(
            episode_size, self.way_num * self.query_num, self.way_num
        )
        # [75, 64, 6, 6], [75, 5, 6, 6]
        output, output = self.cam_layer(
            support_feat, query_feat, support_target_one_hot, query_target_one_hot
        )
        loss1 = self.loss_func(output, query_global_targets.contiguous().view(-1))
        loss2 = self.loss_func(output, query_target.view(-1))
        loss = loss1 + 0.5 * loss2
        output = torch.sum(output.view(*output.size()[:2], -1), dim=-1)  # [300, 5]
        acc = accuracy(output, query_target.view(-1))
        return output, acc, loss
