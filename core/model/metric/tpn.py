"""
@misc{liu2019learningpropagatelabelstransductive,
      title={Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning},
      author={Yanbin Liu and Juho Lee and Minseop Park and Saehoon Kim and Eunho Yang and Sung Ju Hwang and Yi Yang},
      year={2019},
      eprint={1805.10002},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1805.10002},
}

Adapted From https://github.com/csyanbin/TPN-pytorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metric_model import MetricModel


class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1),
        )

        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x, rn):
        x = x.view(-1, 64, 5, 5)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out


class TPN(MetricModel):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)

        self.relation = RelationNetwork()

        if self.rn == 300:
            self.alpha = torch.tensor([alpha], requires_grad=False).to(self.device)
        elif self.rn == 30:
            self.alpha = nn.Parameter(
                torch.tensor([alpha]).to(self.device), requires_grad=True
            )

    def labels_to_onehot(self, labels):
        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, self.way_num).to(self.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        return one_hot

    def label_propagation(self, support, query, support_label, query_label):
        eps = np.finfo(float).eps

        input_feat = torch.cat((support, query), 0)
        embedding_all = self.emb_func(input_feat).view(-1, 1600)
        num_nodes = embedding_all.shape[0]

        if self.rn in [30, 300]:
            self.sigma = self.relation(embedding_all, self.rn)
            embedding_all = embedding_all / (self.sigma + eps)
            embedding_1 = torch.unsqueeze(embedding_all, 1)
            embedding_2 = torch.unsqueeze(embedding_all, 0)
            weight_matrix = ((embedding_1 - embedding_2) ** 2).mean(2)
            weight_matrix = torch.exp(-weight_matrix / 2)

        if self.topk > 0:
            topk_values, topk_indices = torch.topk(weight_matrix, self.topk)
            mask = torch.zeros_like(weight_matrix)
            mask = mask.scatter(1, topk_indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
            weight_matrix = weight_matrix * mask

        degree_matrix = weight_matrix.sum(0)
        degree_sqrt_inv = torch.sqrt(1.0 / (degree_matrix + eps))
        degree_1 = torch.unsqueeze(degree_sqrt_inv, 1).repeat(1, num_nodes)
        degree_2 = torch.unsqueeze(degree_sqrt_inv, 0).repeat(num_nodes, 1)
        symmetric_matrix = degree_1 * weight_matrix * degree_2

        support_labels = support_label
        unlabeled_query = torch.zeros(self.way_num * self.query_num, self.way_num).to(
            self.device
        )
        combined_labels = torch.cat((support_labels, unlabeled_query), 0)
        propagated_labels = torch.matmul(
            torch.inverse(
                torch.eye(num_nodes).to(self.device)
                - self.alpha * symmetric_matrix
                + eps
            ),
            combined_labels,
        )
        query_predictions = propagated_labels[self.way_num * self.shot_num :, :]

        ground_truth = torch.argmax(torch.cat((support_label, query_label), 0), 1)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(propagated_labels, ground_truth)

        predicted_query = torch.argmax(query_predictions, 1)
        ground_truth_query = torch.argmax(query_label, 1)
        correct_predictions = (predicted_query == ground_truth_query).sum()
        total_queries = self.query_num * self.way_num
        accuracy = 1.0 * correct_predictions.float() / float(total_queries)

        accuracy = torch.tensor([accuracy])

        return loss, accuracy

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )

        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)

        loss_list = []
        acc_list = []

        for i in range(episode_size):
            support_label_onehot = self.labels_to_onehot(support_target[i])
            query_label_onehot = self.labels_to_onehot(query_target[i])

            loss, acc = self.label_propagation(
                support_image[i],
                query_image[i],
                support_label_onehot,
                query_label_onehot,
            )

            loss_list.append(loss)
            acc_list.append(acc)

        loss = torch.stack(loss_list)
        loss = torch.mean(loss)
        acc = torch.stack(acc_list)
        acc = torch.mean(acc) * 100.0

        return None, acc, loss

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )

        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)

        acc_list = []

        for i in range(episode_size):
            support_label_onehot = self.labels_to_onehot(support_target[i])
            query_label_onehot = self.labels_to_onehot(query_target[i])

            _, acc = self.label_propagation(
                support_image[i],
                query_image[i],
                support_label_onehot,
                query_label_onehot,
            )

            acc_list.append(acc)

        acc = torch.stack(acc_list)
        acc = torch.mean(acc) * 100.0

        return None, acc
