import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class DN4(MetricModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, n_k=3):
        super(DN4, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.n_k = n_k
        self.loss_func = nn.CrossEntropyLoss()
        self._init_network()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        query_images, query_targets, support_images, _ = batch
        query_images = torch.cat(query_images, 0)
        query_targets = torch.cat(query_targets, 0)
        support_images = torch.cat(support_images, 0)
        query_images = query_images.to(self.device)
        query_targets = query_targets.to(self.device)
        support_images = support_images.to(self.device)

        query_feat = self.model_func(query_images)
        support_feat = self.model_func(support_images)

        output = self.calc_image2class(query_feat, support_feat)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))
        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        query_images, query_targets, support_images, _ = batch
        query_images = torch.cat(query_images, 0)
        query_targets = torch.cat(query_targets, 0)
        support_images = torch.cat(support_images, 0)
        query_images = query_images.to(self.device)
        query_targets = query_targets.to(self.device)
        support_images = support_images.to(self.device)

        query_feat = self.model_func(query_images)
        support_feat = self.model_func(support_images)

        output = self.calc_image2class(query_feat, support_feat)
        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))
        return output, prec1, loss

    def calc_image2class(self, input1, input2):
        """

        :param input1: (query_num * way_num, feat_dim, feat_width, feat_height)
        :param input2: (support_num * way_num, feat_dim, feat_width, feat_height)
        :return: query_num * way_num * way_num, feat_dim, feat_width, feat_height
        """
        feat_dim = input1.size(1)

        # way_num * query_num, 1, feat_width * feat_height, feat_dim
        input1 = input1.view(self.way_num * self.query_num, feat_dim, -1).permute(0, 2, 1)
        input1 = F.normalize(input1, p=2, dim=2).unsqueeze(1)

        # way_num, feat_dim, shot_num * feat_width * feat_height
        input2 = input2.view(self.way_num, self.shot_num, feat_dim, -1) \
            .permute(0, 2, 1, 3).contiguous().view(self.way_num, feat_dim, -1)
        input2 = F.normalize(input2, p=1, dim=1)

        # way_num * query_num, way_num, feat_width * feat_height,
        # shot_num * feat_width * feat_height
        relation = torch.matmul(input1, input2)
        topk_value, _ = torch.topk(relation, self.n_k)
        score = torch.sum(topk_value, dim=[2, 3])

        return score
