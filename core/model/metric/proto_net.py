import torch
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class Proto_Layer(nn.Module):
    def __init__(self, train_way, train_shot, train_query):
        super(Proto_Layer, self).__init__()
        self.train_way = train_way
        self.train_shot = train_shot
        self.train_query = train_query

    def forward(self, query_feat, support_feat):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # t, wq, c
        query_feat = query_feat.view(t, self.train_way * self.train_query, c)
        # t, w, c
        support_feat = support_feat.view(t, self.train_way, self.train_shot, c)
        proto_feat = torch.mean(support_feat, dim=2)

        # t, wq, 1, c - t, 1, w, c -> t, wq, w
        dist = -torch.sum(torch.pow(query_feat.unsqueeze(2) -
                                    proto_feat.unsqueeze(1), 2), dim=3)

        return dist


class ProtoNet(MetricModel):
    def __init__(self, train_way, train_shot, train_query, emb_func, device):
        super(ProtoNet, self).__init__(train_way, train_shot, train_query, emb_func, device)
        self.proto_layer = Proto_Layer(train_way, train_shot, train_query)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.train_way * (self.train_shot + self.train_query))
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat,mode=1)

        output = self.proto_layer(query_feat, support_feat) \
            .view(episode_size * self.train_way * self.train_query, self.train_way)
        acc = accuracy(output, query_target)

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (self.train_way * (self.train_shot + self.train_query))
        emb = self.emb_func(images)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(emb,mode=1)

        output = self.proto_layer(query_feat, support_feat) \
            .view(episode_size * self.train_way * self.train_query, self.train_way)
        loss = self.loss_func(output, query_target)
        acc = accuracy(output, query_target)

        return output, acc, loss
