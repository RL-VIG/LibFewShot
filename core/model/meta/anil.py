import copy

import torch
from torch import nn
import torch.nn.functional as F

from core.utils import accuracy
from .meta_model import MetaModel

from ..backbone.maml_backbone import Linear_fw

# FIXME 方法类初始不需要赋值

class ANIL_Layer(nn.Module):
    def __init__(self, feat_dim, hid_dim, way_num):
        super(ANIL_Layer, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(feat_dim, hid_dim)
            Linear_fw(feat_dim, way_num)
        )

    def forward(self, x):
        return self.layers(x)


class ANIL(MetaModel):
    def __init__(self, way_num, shot_num, query_num, emb_func, device, inner_para, feat_dim, hid_dim):
        super(ANIL, self).__init__(way_num, shot_num, query_num, emb_func, device)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = ANIL_Layer(feat_dim=feat_dim, hid_dim=hid_dim, way_num=way_num)
        self.inner_para = inner_para
        self._init_network()

    def set_forward(self, batch, ):
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            self.test_loop(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc, _ = accuracy(output.squeeze(), query_target.contiguous().reshape(-1), topk=(1, 3))
        return output, acc

    def set_forward_loss(self, batch, ):
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            self.train_loop(support_feat[i], support_target[i])
            output = self.classifier(query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().reshape(-1))
        acc, _ = accuracy(output.squeeze(), query_target.contiguous().reshape(-1), topk=(1, 3))
        return output, acc, loss

    def train_loop(self, support_feat, support_target):
        return self.set_forward_adaptation(support_feat, support_target)

    def test_loop(self, support_feat, support_target):
        return self.set_forward_adaptation(support_feat, support_target)

    def set_forward_adaptation(self, support_feat, support_target):
        lr = self.inner_para['lr']
        fast_parameters = list(self.classifier.parameters())
        for parameter in self.classifier.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()

        for i in range(self.inner_para['iter']):
            output = self.classifier(support_feat)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.classifier.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - self.inner_para['lr'] * grad[k]
                fast_parameters.append(weight.fast)
