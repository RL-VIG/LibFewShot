import torch
from torch import digamma, nn
import torch.nn.functional as F
import copy

from core.model.abstract_model import AbstractModel
from core.utils import accuracy
from .meta_model import MetaModel

from ..backbone.maml_backbone import Linear_fw

# TODO, refer

class MAML_Layer(nn.Module):
    def __init__(self, feat_dim=64, train_way=5) -> None:
        super(MAML_Layer, self).__init__()
        self.layers = nn.Sequential(
            Linear_fw(feat_dim, train_way)
        )

    def forward(self, x):
        return self.layers(x)


class MAML(MetaModel):
    def __init__(self, train_way, train_shot, train_query, feature, device, inner_para, feat_dim):
        super(MAML, self).__init__(train_way, train_shot, train_query, feature, device)
        self.feat_dim = feat_dim
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = MAML_Layer(feat_dim, train_way=train_way)
        self.inner_para = inner_para

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def set_forward(self, batch, ):
        image, global_target = batch # unused global_target
        image = image.to(self.device)
        support_image, query_image, support_target, query_target = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_target = query_target[i].reshape(-1)
            self.train_loop(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch, ):
        image, global_target = batch # unused global_target
        image = image.to(self.device)
        support_image, query_image, support_target, query_target = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            # episode_query_targets = query_targets[i].reshape(-1)
            self.train_loop(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def train_loop(self, support_set, support_target):
        return self.set_forward_adaptation(support_set, support_target)

    def test_loop(self, support_set, support_target):
        return self.set_forward_adaptation(support_set, support_target)

    def set_forward_adaptation(self, support_set, support_target):
        lr = self.inner_para['lr']
        fast_parameters = list(self.parameters())
        for parameter in self.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()
        for i in range(self.inner_para['iter']):
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - self.inner_para['lr'] * grad[k]
                fast_parameters.append(weight.fast)
