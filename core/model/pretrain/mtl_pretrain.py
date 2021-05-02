import torch
from torch import nn

from core.utils import accuracy
from .pretrain_model import PretrainModel
import torch.nn.functional as F

# FIXME ways改为way_num
# adapted from https://github.com/yaoyao-liu/meta-transfer-learning
class MTLBaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, ways, z_dim):
        super().__init__()
        self.ways = ways
        self.z_dim = z_dim
        self.var = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.ways, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.var.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.ways))
        self.var.append(self.fc1_b)

    def forward(self, input_x, the_var=None):
        if the_var is None:
            the_var = self.var
        fc1_w = the_var[0]
        fc1_b = the_var[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars
        

class MTLPretrain(PretrainModel): # use image-size=80 in repo
    def __init__(self, way_num, shot_num, query_num, emb_func, device, feat_dim,
                 num_class,inner_train_iter):
        super(MTLPretrain, self).__init__(way_num, shot_num, query_num, emb_func, device)
        self.feat_dim = feat_dim
        self.num_class = num_class

        self.pre_fc = nn.Sequential(nn.Linear(self.feat_dim, 1000), nn.ReLU(), nn.Linear(1000, self.num_class))
        self.base_learner = MTLBaseLearner(way_num,z_dim=self.feat_dim)
        self.inner_train_iter=inner_train_iter

        self.loss_func = nn.CrossEntropyLoss()


    def set_forward(self, batch, ):
        """
        meta-validation
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device)

        with torch.no_grad():
            feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=4)

        classifier, fast_weight = self.test_loop(support_feat, support_target)

        output = classifier(query_feat,fast_weight)

        acc, _ = accuracy(output, query_target, topk=(1, 3))

        return output, acc

    def set_forward_loss(self, batch):
        """
        pretrain
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device).contiguous()

        feat = self.emb_func(image)

        output = self.pre_fc(feat).contiguous()

        loss = self.loss_func(output, global_target)
        acc, _ = accuracy(output, global_target, topk=(1, 3))
        return output, acc, loss

    def test_loop(self, support_feat, support_target):
        return self.set_forward_adaptation(support_feat, support_target)

    def set_forward_adaptation(self, support_feat, support_target):

        classifier = self.base_learner.to(self.device)

        logit = self.base_learner(support_feat)
        loss = self.loss_func(logit,support_target)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_parameters = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))

        for _ in range(1, self.inner_train_iter):
            logit = self.base_learner(support_feat, fast_parameters)
            loss = F.cross_entropy(logit, support_target)
            grad = torch.autograd.grad(loss, fast_parameters)
            fast_parameters = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_parameters)))

        return classifier, fast_parameters