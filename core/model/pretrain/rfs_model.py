import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .pretrain_model import PretrainModel


class RFSModel(PretrainModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim,
                 num_classes, inner_optim=None, inner_train_iter=20, ):
        super(RFSModel, self).__init__(way_num, shot_num, query_num, model_func, device)

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.inner_optim = inner_optim
        self.inner_train_iter = inner_train_iter

        self.classifier = nn.Linear(self.feat_dim, self.num_classes)
        self.loss_func = nn.CrossEntropyLoss()

        self._init_network()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        with torch.no_grad():
            emb = self.model_func(images)
        support_feat, query_feat, support_targets, query_targets = self.split_by_episode(emb,mode=4)

        classifier = self.test_loop(support_feat, support_targets)

        query_feat = F.normalize(query_feat, p=2, dim=1).detach().cpu().numpy()
        query_targets = query_targets.detach().cpu().numpy()

        output = classifier.predict(query_feat)
        prec1 = metrics.accuracy_score(query_targets, output) * 100

        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        feat = self.model_func(images)
        output = self.classifier(feat)
        loss = self.loss_func(output, targets)

        prec1, _ = accuracy(output, targets, topk=(1, 3))

        return output, prec1, loss

    def test_loop(self, support_feat, support_targets):
        return self.set_forward_adaptation(support_feat, support_targets)

    def set_forward_adaptation(self, support_feat, support_targets):
        classifier = LogisticRegression(penalty='l2',
                                        random_state=0,
                                        C=1.0,
                                        solver='lbfgs',
                                        max_iter=1000,
                                        multi_class='multinomial')

        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_targets = support_targets.detach().cpu().numpy()

        classifier.fit(support_feat, support_targets)

        return classifier
