import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class DN4_Layer(nn.Module):
    def __init__(self, train_way, train_shot, train_query, n_k):
        super(DN4_Layer, self).__init__()
        self.train_way = train_way
        self.train_shot = train_shot
        self.train_query = train_query
        self.n_k = n_k

    def forward(self, query_feat, support_feat):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        query_feat = query_feat.view(t, self.train_way * self.train_query, c, h * w) \
            .permute(0, 1, 3, 2)
        query_feat = F.normalize(query_feat, p=2, dim=2).unsqueeze(2)

        # t, ws, c, h, w -> t, w, s, c, hw -> t, 1, w, c, shw
        support_feat = support_feat.view(t, self.train_way, self.train_shot, c, h * w) \
            .permute(0, 1, 3, 2, 4).contiguous() \
            .view(t, self.train_way, c, self.train_shot * h * w)
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)

        # t, wq, w, hw, shw -> t, wq, w, hw, n_k -> t, wq, w
        relation = torch.matmul(query_feat, support_feat)
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        score = torch.sum(topk_value, dim=[3, 4])

        return score


class DN4(MetricModel):
    def __init__(self, train_way, train_shot, train_query, emb_func, device, n_k=3):
        super(DN4, self).__init__(train_way, train_shot, train_query, emb_func, device)
        self.dn4_layer = DN4_Layer(train_way, train_shot, train_query, n_k)
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
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat,mode=2)

        output = self.dn4_layer(query_feat, support_feat) \
            .view(episode_size * self.train_way * self.train_query, self.train_way)
        acc = accuracy(output, query_target)

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.train_way * (self.train_shot + self.train_query))
        emb = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(emb,mode=2)

        output = self.dn4_layer(query_feat, support_feat) \
            .view(episode_size * self.train_way * self.train_query, self.train_way)
        loss = self.loss_func(output, query_target)
        acc = accuracy(output, query_target)

        return output, acc, loss
