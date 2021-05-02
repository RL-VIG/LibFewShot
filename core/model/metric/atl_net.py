import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class AEAModule(nn.Module):
    def __init__(self, inplanes, scale_value, from_value, value_interval):
        super(AEAModule, self).__init__()

        self.inplanes = inplanes
        self.scale_value = scale_value
        self.from_value = from_value
        self.value_interval = value_interval

        self.f_psi = nn.Sequential(
            nn.Linear(self.inplanes, self.inplanes // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.inplanes // 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, f_x):
        # f_x -> t, wq, hw, wshw
        t, wq, hw, c = x.size()

        # t, wq, hw, c -> t, wq, hw, 1
        clamp_value = self.f_psi(x.view(t * wq * hw, c)) * self.value_interval + self.from_value
        clamp_value = clamp_value.view(t, wq, hw, 1)
        clamp_fx = torch.sigmoid(self.scale_value * (f_x - clamp_value))
        attention_mask = F.normalize(clamp_fx, p=1, dim=-1)

        return attention_mask


class ATLLayer(nn.Module):
    def __init__(self, way_num, shot_num, query_num, inplanes,
                 scale_value, atten_scale_value, from_value, value_interval):
        super(ATLLayer, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.inplanes = inplanes
        self.scale_value = scale_value
        self.atten_scale_value = atten_scale_value
        self.from_value = from_value
        self.value_interval = value_interval

        self.W = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.atten_layer = AEAModule(self.inplanes, self.atten_scale_value, self.from_value, self.value_interval)

    def forward(self, query_feat, support_feat):
        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c
        # t, ws, c, hw -> t, c, ws, hw -> t, 1, c, wshw
        w_query = self.W(query_feat.view(t * wq, c, h, w)).view(t, wq, c, h * w).permute(0, 1, 3, 2).contiguous()
        w_support = self.W(support_feat.view(t * ws, c, h, w)).view(t, ws, c, h * w) \
            .permute(0, 2, 1, 3).contiguous().view(t, 1, c, ws * h * w)

        w_query = F.normalize(w_query, dim=3)
        w_support = F.normalize(w_support, dim=2)

        # t, wq, hw, c matmul t, 1, c, wshw -> t, wq, hw, wshw
        f_x = torch.matmul(w_query, w_support)
        atten_score = self.atten_layer(w_query, f_x)

        # t, wq, c, hw -> t, wq, hw, c
        # t, ws, c, hw -> t, c, ws, hw -> t, 1, c, wshw
        query_feat = query_feat.view(t, wq, c, h * w).permute(0, 1, 3, 2).contiguous()
        support_feat = support_feat.view(t, ws, c, h * w).permute(0, 2, 1, 3).contiguous().view(t, 1, c, ws * h * w)

        query_feat = F.normalize(query_feat, dim=3)
        support_feat = F.normalize(support_feat, dim=2)

        # t, wq, hw, c matmul t, 1, c, wshw -> t, wq, hw, wshw
        # t, wq, hw, wshw -> t, wq, hw, w, s, hw -> t, wq, w, s, hw, hw -> t, wq, w
        match_score = torch.matmul(query_feat, support_feat)

        atten_match_score = torch.mul(atten_score, match_score) \
            .view(t, wq, h * w, self.way_num, self.shot_num, h * w).permute(0, 1, 3, 4, 2, 5)
        score = torch.sum(atten_match_score, dim=5)
        score = torch.mean(score, dim=[3, 4]) * self.scale_value

        return score


# TODO a large gap in the 5-way 5-shot
class ATLNet(MetricModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, inplanes, scale_value=30,
                 atten_scale_value=50, from_value=0.5, value_interval=0.3):
        super(ATLNet, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.atl_layer = ATLLayer(way_num, shot_num, query_num, inplanes, scale_value,
                                  atten_scale_value, from_value, value_interval)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (self.way_num * (self.shot_num + self.query_num))
        feat = self.emb_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(feat, mode=2)

        output = self.atl_layer(query_feat, support_feat) \
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
        emb = self.emb_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(emb, mode=2)

        output = self.atl_layer(query_feat, support_feat) \
            .view(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1, loss
