import torch
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class ProtoNet(MetricModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device):
        super(ProtoNet, self).__init__(way_num, shot_num, query_num, model_func, device)
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
        proto_feat = torch.mean(support_feat.view(self.way_num, self.shot_num, -1), 1)

        output = self.euclidean_dist(query_feat, proto_feat)
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
        proto_feat = torch.mean(support_feat.view(self.way_num, self.shot_num, -1), 1)

        output = self.euclidean_dist(query_feat, proto_feat)
        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))
        return output, prec1, loss

    def train_loop(self, *args, **kwargs):
        raise NotImplementedError

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError

    def euclidean_dist(self, input1, input2):
        """
        calc euclidean distance for each of the input1 to input2
        :param input1: [query_num * way_num, feat_dim]
        :param input2: [way_num, feat_dim]
        :return:
        """
        input1 = input1.unsqueeze(1)
        input2 = input2.unsqueeze(0)

        dist = -torch.sum(torch.pow(input1 - input2, 2), 2)

        return dist
