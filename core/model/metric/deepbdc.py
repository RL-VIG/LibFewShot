import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from core.model.metric.metric_model import MetricModel


"""
@inproceedings{DeepBDC-CVPR2022,
    title={Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification},
    author={Jiangtao Xie and Fei Long and Jiaming Lv and Qilong Wang and Peihua Li}, 
    booktitle={CVPR},
    year={2022}
 }
"""


class ProtoLayer(nn.Module):
    """
    This Proto Layer is partly different from Proto_layer @ ProtoNet
    """
    def __init__(self):
        super(ProtoLayer, self).__init__()
    
    def forward(self, query_feat, support_feat, way_num, shot_num, query_num):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # t, wq, c
        query_feat = query_feat.reshape(t, way_num * query_num, c)
        # t, w, c -- proto_feat
        support_feat = support_feat.reshape(t, way_num, shot_num, c)
        proto_feat = torch.mean(support_feat, dim=2)

        if shot_num > 1:
            # euclidean, 
            # t, wq, 1, d - t, 1, w, d -> t, wq, w
            return (lambda x, y: -torch.sum(
                torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),
                dim=3,
            ))(query_feat, proto_feat)
        else:
            # cosine similarity
            # t, wq, d - t, d, w -> t, wq, w
            return (lambda x, y: torch.matmul(
                # F.normalize(x, p=2, dim=-1),
                # torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)
                # DeepBDC did not normalize the query_feat and proto_feat
                x,
                torch.transpose(y, -1, -2)
            ))(query_feat, proto_feat)


class DeepBDC(MetricModel):
    """ 
    This class is modified from ProtoNet @ ProtoNet
    """
    def __init__(self, **kwargs):
        super(DeepBDC, self).__init__(**kwargs)
        self.proto_layer = ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()
    
    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)     # [bsize, c] -- [bsize = t * way_num * (n_s + n_q), c=r*(r+1)/2

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        # support_feat -- [t, ws, c], query_feat -- [t, wq, c], 
        # support_target -- [t, ws],        query_feat -- [t, wq]

        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)
        # output expected be [t, wq, w] ---> [t*w*q, w]
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc


    def set_forward_loss(self, batch):
        images, global_target = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(images)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc, loss
