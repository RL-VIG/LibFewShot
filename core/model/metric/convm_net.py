import torch
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class ConvMLayer(nn.Module):
    def __init__(self, way_num, shot_num, query_num, n_local):
        super(ConvMLayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num

        self.conv1d_layer = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),
            nn.Conv1d(in_channels=1, out_channels=1,
                      kernel_size=n_local, stride=n_local)
        )

    def _calc_support_cov(self, support_feat):
        t, ws, c, h, w = support_feat.size()

        # t, ws, c, h, w -> t, ws, hw, c -> t, w, shw, c
        support_feat = support_feat.view(t, ws, c, h * w).permute(0, 1, 3, 2).contiguous()
        support_feat = support_feat.view(t, self.way_num, self.shot_num * h * w, c)
        support_feat = support_feat - torch.mean(support_feat, dim=2, keepdim=True)

        # t, w, c, c
        cov_mat = torch.matmul(support_feat.permute(0, 1, 3, 2), support_feat)
        cov_mat = torch.div(cov_mat, h * w - 1)

        return cov_mat

    def _calc_similarity(self, query_feat, support_cov_mat):
        t, wq, c, h, w = query_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        query_feat = query_feat.view(t, wq, c, h * w).permute(0, 1, 3, 2).contiguous()
        query_feat = query_feat - torch.mean(query_feat, dim=2, keepdim=True)
        query_feat = query_feat.unsqueeze(2)

        # t, wq, 1, hw, c matmul t, 1, w, c, c -> t, wq, w, hw, c
        # t, wq, w, hw, c matmul t, wq, 1, c, hw -> t, wq, w, hw, hw -> twqw, hw, hw
        support_cov_mat = support_cov_mat.unsqueeze(1)
        prod_mat = torch.matmul(query_feat, support_cov_mat)
        prod_mat = torch.matmul(prod_mat, torch.transpose(query_feat, 3, 4)) \
            .contiguous().view(t * self.way_num * wq, h * w, h * w)

        # twq, 1, whw
        cov_sim = torch.diagonal(prod_mat, dim1=1, dim2=2).contiguous()
        cov_sim = cov_sim.view(t * wq, 1, self.way_num * h * w)

        return cov_sim

    def forward(self, query_feat, support_feat):
        t, wq, c, h, w = query_feat.size()
        support_cov_mat = self._calc_support_cov(support_feat)
        cov_sim = self._calc_similarity(query_feat, support_cov_mat)
        score = self.conv1d_layer(cov_sim).view(t, wq, self.way_num)

        return score


class ConvMNet(MetricModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, n_local=3):
        super(ConvMNet, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.convm_layer = ConvMLayer(way_num, shot_num, query_num, n_local)
        self.loss_func = nn.CrossEntropyLoss()
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
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(emb,mode=2)

        output = self.convm_layer(query_feat, support_feat) \
            .view(episode_size * self.way_num * self.query_num, self.way_num)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        emb = self.model_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(emb,mode=2)

        output = self.convm_layer(query_feat, support_feat) \
            .view(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1, loss
