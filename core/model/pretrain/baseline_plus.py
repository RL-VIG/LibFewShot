import torch
from torch import nn
from torch.nn.utils import weight_norm

from core.utils import accuracy
from .pretrain_model import PretrainModel


class DistLinear(nn.Module):
    """
    Coming from "A Closer Look at Few-shot Classification. ICLR 2019."
    https://github.com/wyharveychen/CloserLookFewShot.git
    """

    def __init__(self, in_channel, out_channel):
        super(DistLinear, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel, bias=False)
        # See the issue#4&8 in the github
        self.class_wise_learnable_norm = True
        # split the weight update component to direction and norm
        if self.class_wise_learnable_norm:
            weight_norm(self.fc, 'weight', dim=0)

        if out_channel <= 200:
            # a fixed scale factor to scale the output of cos value
            # into a reasonably large input for softmax
            self.scale_factor = 2
        else:
            # in omniglot, a larger scale factor is
            # required to handle >1000 output classes.
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            fc_norm = torch.norm(self.fc.weight.data, p=2, dim=1) \
                .unsqueeze(1).expand_as(self.fc.weight.data)
            self.fc.weight.data = self.fc.weight.data.div(fc_norm + 0.00001)
        # matrix product by forward function, but when using WeightNorm,
        # this also multiply the cosine distance by a class-wise learnable norm
        cos_dist = self.fc(x_normalized)
        scores = self.scale_factor * cos_dist

        return scores


class BaselinePlus(PretrainModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim,
                 num_classes, inner_optim=None, inner_train_iter=20):
        super(BaselinePlus, self).__init__(way_num, shot_num, query_num, model_func,
                                           device)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.inner_optim = inner_optim
        self.inner_train_iter = inner_train_iter

        self.loss_func = nn.CrossEntropyLoss()

        self._init_network()

        self.classifier = DistLinear(self.feat_dim, self.num_classes)

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        with torch.no_grad():
            emb = self.emb_func(images)

        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(emb,mode=4)

        classifier = self.test_loop(support_feat, support_targets)

        output = classifier(query_feat)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        feat = self.emb_func(images)
        output = self.classifier(feat)
        loss = self.loss_func(output, targets)
        prec1, _ = accuracy(output, targets, topk=(1, 3))
        return output, prec1, loss

    def test_loop(self, support_feat, support_targets):
        return self.set_forward_adaptation(support_feat, support_targets)

    def set_forward_adaptation(self, support_feat, support_targets):
        classifier = DistLinear(self.feat_dim, self.way_num)
        optimizer = self.sub_optimizer(classifier, self.inner_optim)

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)
        for epoch in range(self.inner_train_iter):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_batch_size):
                select_id = rand_id[i:min(i + self.inner_batch_size, support_size)]
                batch = support_feat[select_id]
                target = support_targets[select_id]

                output = classifier(batch)

                loss = self.loss_func(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return classifier
