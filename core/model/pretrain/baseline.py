import torch
from torch import nn

from core.utils import accuracy
from .pretrain_model import PretrainModel

# FIXME test_loop和train_loop形式要一样
# https://github.com/wyharveychen/CloserLookFewShot.git
# FIXME 加上多GPU


class Baseline(PretrainModel):
    def __init__(
        self,
        way_num,
        shot_num,
        query_num,
        emb_func,
        device,
        feat_dim,
        num_class,
        inner_optim=None,
        inner_batch_size=4,
        inner_train_iter=20,
    ):
        super(Baseline, self).__init__(way_num, shot_num, query_num, emb_func, device)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_optim = inner_optim
        self.inner_batch_size = inner_batch_size
        self.inner_train_iter = inner_train_iter

        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=4
        )

        classifier = self.test_loop(support_feat, support_target)

        output = classifier(query_feat)
        acc = accuracy(output, query_target)

        return output, acc

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss

    def test_loop(self, support_feat, support_target):
        return self.set_forward_adaptation(support_feat, support_target)

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = nn.Linear(self.feat_dim, self.way_num)
        optimizer = self.sub_optimizer(classifier, self.inner_optim)

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)
        for epoch in range(self.inner_train_iter):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_batch_size):
                select_id = rand_id[i : min(i + self.inner_batch_size, support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]

                output = classifier(batch)

                loss = self.loss_func(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return classifier
