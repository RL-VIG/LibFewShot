import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class ProtoLayer(nn.Module):
    def __init__(self, ):
        super(ProtoLayer, self).__init__()

    def forward(self, query_feat, support_feat, way_num, shot_num, query_num):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # t, wq, c
        query_feat = query_feat.view(t, way_num * query_num, c)
        query_feat = F.normalize(query_feat, dim=2, p=2)
        # t, w, c
        support_feat = support_feat.view(t, way_num, shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)
        proto_feat = F.normalize(proto_feat, dim=2, p=2)

        # t, wq, 1, c - t, 1, w, c -> t, wq, w
        dist = -torch.sum(torch.pow(query_feat.unsqueeze(2) -
                                    proto_feat.unsqueeze(1), 2), dim=3)

        return dist


class ProtoNetGlobal(MetricModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim,
                 num_classes, gamma=1.0, test_way=5, test_shot=1, test_query=15):
        super(ProtoNetGlobal, self).__init__(
            way_num, shot_num, query_num, model_func, device)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.gamma = gamma
        self.test_way = test_way
        self.test_shot = test_shot
        self.test_query = test_query
        self.proto_layer = ProtoLayer()
        self.classifier = nn.Linear(self.feat_dim, self.num_classes)
        self.loss_func = nn.CrossEntropyLoss()
        self._init_network()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        feat = self.model_func(images)

        episode_size = images.size(0) // (
                self.test_way * (self.test_shot + self.test_query))
        local_labels = torch.arange(self.test_way, dtype=torch.long).view(1, -1, 1) \
            .repeat(episode_size, 1, self.test_shot + self.test_query).view(-1) \
            .to(self.device).contiguous() \
            .view(episode_size, self.test_way, self.test_shot + self.test_query)

        feat = feat.view(episode_size, self.test_way, self.test_shot + self.test_query,
                         -1)
        support_feat = feat[:, :, :self.test_shot, :].contiguous() \
            .view(episode_size, self.test_way * self.test_shot, -1)
        query_feat = feat[:, :, self.test_shot:, :].contiguous() \
            .view(episode_size, self.test_way * self.test_query, -1)
        support_targets = local_labels[:, :, :self.test_shot].contiguous() \
            .view(episode_size, -1)
        query_targets = local_labels[:, :, self.test_shot:].contiguous() \
            .view(episode_size, -1)

        output = self.proto_layer(query_feat, support_feat,
                                  self.test_way, self.test_shot, self.test_query) \
            .view(episode_size * self.test_way * self.test_query, self.test_way)
        prec1, _ = accuracy(output, query_targets.view(-1), topk=(1, 3))

        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        global_targets = global_targets.to(self.device)
        images = images.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.model_func(images)
        global_output = self.classifier(feat)
        global_loss = self.loss_func(global_output, global_targets.view(-1))

        support_feat, query_feat, support_targets, query_targets = \
            self.split_by_episode(feat, mode=1)

        local_output = self.proto_layer(query_feat, support_feat,
                                        self.way_num, self.shot_num, self.query_num) \
            .view(episode_size * self.way_num * self.query_num, self.way_num)
        local_loss = self.loss_func(local_output, query_targets.view(-1))

        loss = (local_loss + self.gamma * global_loss) / (1 + self.gamma)
        prec1, _ = accuracy(local_output, query_targets.view(-1), topk=(1, 3))

        return local_output, prec1, loss
