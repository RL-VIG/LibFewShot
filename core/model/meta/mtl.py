import torch
from torch import digamma, nn
import torch.nn.functional as F
import copy

from core.model.abstract_model import AbstractModel
from core.utils import accuracy
from .meta_model import MetaModel

class MTLBaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, ways, z_dim):
        super().__init__()
        self.ways = ways
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.ways, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.ways))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

class MTL(MetaModel):
    def __init__(self, way_num, shot_num, query_num, emb_func, device, feat_dim,
                 num_classes, inner_para):
        super(MTL, self).__init__(way_num, shot_num, query_num, emb_func, device)
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.base_learner = MTLBaseLearner(way_num, z_dim=self.feat_dim).to(device)
        self.inner_para = inner_para

        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch, ):
        '''
        meta-validation
        '''
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device)

        with torch.no_grad():
            feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=4)

        classifier, base_learner_weight = self.train_loop(support_feat, support_target)

        output = classifier(query_feat, base_learner_weight)

        acc, _ = accuracy(output, query_target, topk=(1, 3))

        return output, acc

    def set_forward_loss(self, batch, ):
        '''
        meta-train
        '''
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device)

        feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=4)

        classifier, base_learner_weight = self.train_loop(support_feat, support_target)

        output = classifier(query_feat, base_learner_weight)
        loss = self.loss_func(output,query_target)
        acc, _ = accuracy(output, query_target, topk=(1, 3))

        return output, acc, loss

    def train_loop(self, support_feat, support_target):
        return self.set_forward_adaptation(support_feat, support_target)

    def test_loop(self, support_feat, support_target):
        return self.set_forward_adaptation(support_feat, support_target)

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = self.base_learner
        logit = self.base_learner(support_feat)
        loss = self.loss_func(logit, support_target)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_parameters = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))

        for _ in range(1, self.inner_para['iter']):
            logit = self.base_learner(support_feat, fast_parameters)
            loss = F.cross_entropy(logit, support_target)
            grad = torch.autograd.grad(loss, fast_parameters)
            fast_parameters = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_parameters)))

        return classifier, fast_parameters
