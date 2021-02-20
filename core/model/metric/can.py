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
# from .metric_model import MetricModel
# from ..loss import CrossEntropyLoss
from core.model.metric.metric_model import MetricModel
from core.model.loss import CrossEntropyLoss
import pdb

def one_hot(indices, depth, use_cuda=True):
    if use_cuda:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    else:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies

def one_hot_1(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 5
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot

def shuffle(images, targets, global_targets):
    """
    A trick for CAN training
    """
    batch_size, sample_num = images.shape[0], images.shape[1]
    for i in range(4):
        indices = torch.randperm(sample_num).to(images.device)
        images[i] = images[i][indices]
        targets[i] = targets[i][indices]
        global_targets[i] = global_targets[i][indices]
    return images, targets, global_targets


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
    def __init__(self):
        super(CAM, self).__init__()
        self.conv1 = ConvBlock(25, 5, 1)
        self.conv2 = nn.Conv2d(5, 25, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

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
    def __init__(self, scale_cls, iter_num_prob=35.0/75, num_classes=64, nFeat=512):
        super(CAMLayer, self).__init__()
        self.scale_cls = scale_cls
        self.cam = CAM()
        self.iter_num_prob = iter_num_prob
        self.nFeat = nFeat
        self.classifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)

    def val(self, support_feat, query_feat):
        query_feat = query_feat.mean(4)
        query_feat = query_feat.mean(4)
        support_feat = F.normalize(support_feat, p=2, dim=support_feat.dim()-1, eps=1e-12)  # [1, 75, 5, 512]
        query_feat = F.normalize(query_feat, p=2, dim=query_feat.dim()-1, eps=1e-12) # [1, 75, 5, 512]
        scores = self.scale_cls * torch.sum(query_feat * support_feat, dim=-1) # [1, 75, 5]
        return scores

    def forward(self, support_feat, query_feat, support_targets, query_targets):
        """
        support_feat: [4, 5, 512, 6, 6]
        query_feat: [4, 75, 512, 6, 6]
        support_targets: [4, 5, 5] one-hot
        query_targets: [4, 75, 5] one-hot
        """
        original_feat_shape = support_feat.size()
        batch_size = support_feat.size(0)
        n_support = support_feat.size(1)
        n_query = query_feat.size(1)
        way_num = support_targets.size(-1)
        # feat = support_feat.size(-1)

        # flatten feature
        support_feat = support_feat.view(batch_size, n_support, -1)

        labels_train_transposed = support_targets.transpose(1, 2)  # [1, 5, 5]

        # calc the prototypes of support set
        prototypes = torch.bmm(labels_train_transposed, support_feat)  # [1, 5, 5]x[1, 5, 640]
        prototypes = prototypes.div(labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes))  # [1, 5, 640]
        prototypes = prototypes.reshape(batch_size, -1, *original_feat_shape[2:])  # [1, 5, 512, 6, 6]
        prototypes, query_feat = self.cam(prototypes, query_feat) # [1, 75, 5, 512, 6, 6]  # [2, 75, 640, 1, 1]
        prototypes = prototypes.mean(4)
        prototypes = prototypes.mean(4)  # [1, 75, 5, 512]

        if not self.training:
            return self.val(prototypes, query_feat)

        proto_norm = F.normalize(prototypes, p=2, dim=3, eps=1e-12)
        query_norm = F.normalize(query_feat, p=2, dim=3, eps=1e-12)
        proto_norm = proto_norm.unsqueeze(4)
        proto_norm = proto_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(query_norm * proto_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * n_query, *cls_scores.size()[2:])

        query_feat = query_feat.view(batch_size, n_query, way_num, -1)
        query_feat = query_feat.transpose(2, 3)
        query_targets = query_targets.unsqueeze(3)
        query_feat = torch.matmul(query_feat, query_targets)
        query_feat = query_feat.view(batch_size * n_query, -1, *original_feat_shape[-2:])
        query_targets = self.classifier(query_feat)

        return query_targets, cls_scores

    def helper(self, support_feat, query_feat, support_targets):
        """
        support_targets_transposed: one-hot
        """
        b, n, c, h, w = support_feat.size()
        k = support_targets.size(2)

        support_targets_transposed = support_targets.transpose(1, 2)
        support_feat = torch.bmm(support_targets_transposed, support_feat.view(b, n, -1))
        support_feat = support_feat.div(support_targets_transposed.sum(dim=2, keepdim=True).expand_as(support_feat))
        support_feat = support_feat.view(b, -1, c, h, w)

        support_feat, query_feat = self.cam(support_feat, query_feat)
        support_feat = support_feat.mean(-1).mean(-1)
        query_feat = query_feat.mean(-1).mean(-1)

        query_feat = F.normalize(query_feat, p=2, dim=query_feat.dim()-1, eps=1e-12)
        support_feat = F.normalize(support_feat, p=2, dim=support_feat.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(query_feat * support_feat, dim=-1)
        return scores

    def val_transductive(self, support_feat, query_feat, support_targets, query_targets):
        iter_num_prob = self.iter_num_prob
        batch_size, num_train = support_feat.size(0), support_feat.size(1)
        num_test = query_feat.size(1)
        K = support_targets.size(2)

        cls_scores = self.helper(support_feat, query_feat, support_targets)

        num_images_per_iter = int(num_test * iter_num_prob)
        num_iter = num_test // num_images_per_iter

        for i in range(num_iter):
            max_scores, preds = torch.max(cls_scores, 2)
            chose_index = torch.argsort(max_scores.view(-1), descending=True)
            chose_index = chose_index[: num_images_per_iter * (i + 1)]
            ftest_iter = query_feat[0, chose_index].unsqueeze(0)
            preds_iter = preds[0, chose_index].unsqueeze(0)

            preds_iter = one_hot(preds_iter.view(-1), K).cuda()
            preds_iter = preds_iter.view(batch_size, -1, K)

            support_feat_iter = torch.cat((support_feat, ftest_iter), 1)
            support_targets_iter = torch.cat((support_targets, preds_iter), 1)
            cls_scores = self.helper(support_feat_iter, query_feat, support_targets_iter)

        return cls_scores


class CAN(MetricModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, scale_cls, iter_num_prob=35.0/75, num_classes=64, nFeat=512):
        super(CAN, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.cam_layer = CAMLayer(scale_cls, iter_num_prob, num_classes, nFeat)
        self.loss_func = CrossEntropyLoss()
        self._init_network()

    def set_forward(self, batch, ):
        """
        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        emb = self.model_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(emb, mode=2)

        # convert to one-hot
        support_targets_one_hot = one_hot(support_targets.view(episode_size*self.way_num*self.shot_num), self.way_num)
        support_targets_one_hot = support_targets_one_hot.view(episode_size, self.way_num*self.shot_num, self.way_num)
        query_targets_one_hot = one_hot(query_targets.view(episode_size*self.way_num*self.query_num), self.way_num)
        query_targets_one_hot = query_targets_one_hot.view(episode_size, self.way_num*self.query_num, self.way_num)
        # cls_scores = self.cam_layer(support_feat, query_feat, support_targets_one_hot, query_targets_one_hot)
        cls_scores = self.cam_layer.val_transductive(support_feat, query_feat, support_targets_one_hot, query_targets_one_hot)

        cls_scores = cls_scores.view(episode_size * self.way_num*self.query_num, -1)
        prec1, _ = accuracy(cls_scores, query_targets, topk=(1, 3))
        return cls_scores, prec1

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        emb = self.model_func(images)  # [80, 640]
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(emb, mode=2)  # [4,5,512,6,6] [4, 75, 512,6,6] [4, 5] [300]
        support_global_targets, query_global_targets = global_targets[:, :, :self.shot_num], global_targets[:, :, self.shot_num:]
        # TODO: Shuffle label index
        support_feat, support_targets, support_global_targets = shuffle(support_feat, support_targets, support_global_targets.reshape(*support_targets.size()))
        query_feat, query_targets, query_global_targets = shuffle(query_feat, query_targets.reshape(*query_feat.size()[:2]), query_global_targets.reshape(*query_feat.size()[:2]))
        # convert to one-hot
        support_targets_one_hot = one_hot(support_targets.view(episode_size*self.way_num*self.shot_num), self.way_num)
        support_targets_one_hot = support_targets_one_hot.view(episode_size, self.way_num*self.shot_num, self.way_num)
        query_targets_one_hot = one_hot(query_targets.view(episode_size*self.way_num*self.query_num), self.way_num)
        query_targets_one_hot = query_targets_one_hot.view(episode_size, self.way_num*self.query_num, self.way_num)
        # [75, 64, 6, 6], [75, 5, 6, 6]
        output, cls_scores = self.cam_layer(support_feat, query_feat, support_targets_one_hot, query_targets_one_hot)
        loss1 = self.loss_func(output, query_global_targets.contiguous().view(-1))
        loss2 = self.loss_func(cls_scores, query_targets.view(-1))
        loss = loss1 + 0.5 * loss2
        cls_scores = torch.sum(cls_scores.view(*cls_scores.size()[:2], -1), dim=-1)  # [300, 5]
        prec1, _ = accuracy(cls_scores, query_targets.view(-1), topk=(1, 3))
        return output, prec1, loss


if __name__ == '__main__':
    torch.manual_seed(0)

    net = CAMLayer(scale_cls=7)
    net.eval()

    x1 = torch.rand(1, 5, 512, 6, 6)
    x2 = torch.rand(1, 75, 512, 6, 6)
    y1 = torch.rand(1, 5, 5)
    y2 = torch.rand(1, 75, 5)

    # y1 = net.test_transductive(x1, x2, y1, y2)
    y1 = net(x1, x2, y1, y2)
    print(y1.size()) # [1, 75, 5]
    print(y1)
