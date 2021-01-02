import torch
from torch import nn

from core.utils import accuracy
from .pretrain_model import PretrainModel


class RFSModel(PretrainModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, feat_dim,
                 num_classes, loss_gamma=2., inner_optim=None, inner_train_iter=20, ):
        super(RFSModel, self).__init__(way_num, shot_num, query_num, model_func, device)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.loss_gamma = loss_gamma

        self.inner_optim = inner_optim
        self.inner_train_iter = inner_train_iter

        self.cls_classifier = nn.Linear(self.feat_dim, self.num_classes)
        self.rot_classifier = nn.Linear(self.feat_dim, 4)
        self.loss_func = nn.CrossEntropyLoss()

        self._init_network()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        query_images, query_targets, support_images, support_targets = batch
        query_images = torch.cat(query_images, 0)
        query_targets = torch.cat(query_targets, 0)
        support_images = torch.cat(support_images, 0)
        support_targets = torch.cat(support_targets, 0)
        query_images = query_images.to(self.device)
        query_targets = query_targets.to(self.device)
        support_images = support_images.to(self.device)
        support_targets = support_targets.to(self.device)

        with torch.no_grad():
            support_feat = self.model_func(support_images)
            query_feat = self.model_func(query_images)

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
        images = torch.cat(images, 0)
        targets = torch.cat(targets, 0)
        images = images.to(self.device)
        targets = targets.to(self.device)

        generated_images, generated_targets, rot_targets \
            = self.rot_image_generation(images, targets)

        feat = self.model_func(generated_images)
        cls_output = self.cls_classifier(feat)
        rot_output = self.rot_classifier(feat)
        cls_loss = self.loss_func(cls_output, generated_targets)
        rot_loss = self.loss_func(rot_output, rot_targets)
        loss = cls_loss + rot_loss * self.loss_gamma

        prec1, _ = accuracy(cls_output, generated_targets, topk=(1, 3))

        return cls_output, prec1, loss

    def test_loop(self, support_feat, support_targets):
        return self.set_forward_adaptation(support_feat, support_targets)

    def set_forward_adaptation(self, support_feat, support_targets):
        classifier = nn.Linear(self.feat_dim, self.way_num)
        optimizer = self.sub_optimizer(classifier, self.inner_optim)

        classifier = classifier.to(self.device)

        classifier.train()
        for i in range(self.inner_train_iter):
            output = classifier(support_feat)

            loss = self.loss_func(output, support_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return classifier

    def rot_image_generation(self, images, targets):
        batch_size = images.size(0)

        images_90 = images.transpose(2, 3).flip(2)
        images_180 = images.flip(2).flip(3)
        images_270 = images.flip(2).transpose(2, 3)

        generated_images = torch.cat((images, images_90, images_180, images_270), dim=0)
        generated_targets = targets.repeat(4)

        rot_targets = torch.zeros(batch_size * 4)
        rot_targets[batch_size:] += 1
        rot_targets[batch_size * 2:] += 1
        rot_targets[batch_size * 3:] += 1
        rot_targets = rot_targets.long().to(self.device)

        return generated_images, generated_targets, rot_targets
