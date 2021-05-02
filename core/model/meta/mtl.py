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
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim,
                 num_classes, inner_train_iter):
        super(MTL, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.base_learner = MTLBaseLearner(way_num, z_dim=self.feat_dim).to(device)
        self.inner_train_iter = inner_train_iter

        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch, ):
        '''
        meta-validation
        '''
        images, global_targets = batch
        images = images.to(self.device)
        global_targets = global_targets.to(self.device)

        with torch.no_grad():
            feat = self.emb_func(images)

        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(feat, mode=4)

        classifier, fast_weights = self.train_loop(support_feat, support_targets)

        output = classifier(query_feat, fast_weights)

        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1

    def set_forward_loss(self, batch, ):
        '''
        meta-train
        '''
        images, global_targets = batch
        images = images.to(self.device)
        global_targets = global_targets.to(self.device)

        feat = self.emb_func(images)

        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(feat, mode=4)

        classifier, fast_weights = self.train_loop(support_feat, support_targets)

        output = classifier(query_feat, fast_weights)
        loss = self.loss_func(output,query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1, loss

    def train_loop(self, support_feat, support_targets):
        classifier = self.base_learner
        logits = self.base_learner(support_feat)
        loss = self.loss_func(logits, support_targets)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))

        for _ in range(1, self.inner_train_iter):
            logits = self.base_learner(support_feat, fast_weights)
            loss = F.cross_entropy(logits, support_targets)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))

        return classifier, fast_weights

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError
