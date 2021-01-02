import torch
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class RelationLayer(nn.Module):
    def __init__(self, feat_dim=64, feat_height=3, feat_width=3):
        super(RelationLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(feat_dim * feat_height * feat_width, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        out = self.layers(x).view(x.size(0), -1)
        out = self.fc(out)
        return out


class RelationNet(MetricModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim=64,
                 feat_height=3, feat_width=3):
        super(RelationNet, self).__init__(way_num, shot_num, query_num, model_func,
                                          device)
        self.feat_dim = feat_dim
        self.feat_height = feat_height
        self.feat_width = feat_width
        self.relation_layer = RelationLayer(self.feat_dim, self.feat_height,
                                            self.feat_width)
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

        relation_pairs = self.cal_pairs(query_feat, support_feat)
        output = self.relation_layer(relation_pairs).view(-1, self.way_num)

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

        relation_pairs = self.cal_pairs(query_feat, support_feat)
        output = self.relation_layer(relation_pairs).view(-1, self.way_num)

        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))
        return output, prec1, loss

    def cal_pairs(self, input1, input2):
        """

        :param input1: (query_num * way_num, feat_dim, feat_width, feat_height)
        :param input2: (support_num * way_num, feat_dim, feat_width, feat_height)
        :return: query_num * way_num * way_num, feat_dim, feat_width, feat_height
        """
        _, c, h, w = input1.size()
        # query_num * way_num, way_num, feat_dim, feat_width, feat_height
        input1 = input1.unsqueeze(0).repeat(self.way_num, 1, 1, 1, 1)
        input1 = torch.transpose(input1, 0, 1)

        # query_num * way_num, way_num, feat_dim, feat_width, feat_height
        input2 = input2.view(self.way_num, self.shot_num, c, h, w)
        input2 = torch.sum(input2, dim=1).unsqueeze(0) \
            .repeat(self.way_num * self.query_num, 1, 1, 1, 1)

        relation_pairs = torch.cat((input1, input2), dim=2).view(-1, c * 2, h, w)
        return relation_pairs
