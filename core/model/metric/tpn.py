import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .metric_model import MetricModel

class RelationNetwork(nn.Module):
    """Graph Construction Module"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,1,kernel_size=3,padding=1),
                        nn.BatchNorm2d(1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2*2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)            # max-pool without padding 
        self.m1 = nn.MaxPool2d(2, padding=1) # max-pool with padding

    def forward(self, x, rn):
        
        x = x.view(-1,64,5,5)
        
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0),-1) 
        out = F.relu(self.fc3(out))
        out = self.fc4(out) # no relu

        out = out.view(out.size(0),-1) # bs*1

        return out


class TPN(MetricModel):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)

        self.relation = RelationNetwork()

        if self.rn == 300:
            self.alpha = torch.tensor([alpha], requires_grad=False).to(self.device)
        elif self.rn == 30:
            self.alpha = nn.Parameter(torch.tensor([alpha]).to(self.device), requires_grad=True)

    def labels_to_onehot(self, labels):
        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, self.way_num).to(self.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        return one_hot

    def label_propagation(self, support, query, s_label, q_label):
        eps = np.finfo(float).eps

        inp = torch.cat((support, query), 0)
        emb_all = self.emb_func(inp).view(-1, 1600)
        N, d = emb_all.shape[0], emb_all.shape[1]

        if self.rn in [30, 300]:
            self.sigma = self.relation(emb_all, self.rn)
            emb_all = emb_all / (self.sigma + eps)
            emb1 = torch.unsqueeze(emb_all,1)
            emb2 = torch.unsqueeze(emb_all,0)
            W = ((emb1-emb2)**2).mean(2)
            W = torch.exp(-W/2)

        if self.topk > 0:
            topk, indices = torch.topk(W, self.topk)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
            W = W * mask

        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        ys = s_label
        yu = torch.zeros(self.way_num * self.query_num, self.way_num).to(self.device)
        y = torch.cat((ys, yu), 0)
        F = torch.matmul(torch.inverse(torch.eye(N).to(self.device) - self.alpha * S + eps), y)
        Fq = F[self.way_num * self.shot_num:, :]

        gt = torch.argmax(torch.cat((s_label, q_label), 0), 1)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(F, gt)

        predq = torch.argmax(Fq,1)
        gtq = torch.argmax(q_label,1)
        correct = (predq==gtq).sum()
        total   = self.query_num * self.way_num
        acc = 1.0 * correct.float() / float(total)

        acc = torch.tensor([acc])

        return loss, acc


    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))

        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)

        loss_list = []
        acc_list = []

        for i in range(episode_size):
            s_label_onehot = self.labels_to_onehot(support_target[i])
            q_label_onehot = self.labels_to_onehot(query_target[i])
            
            loss, acc = self.label_propagation(support_image[i], query_image[i], s_label_onehot, q_label_onehot)

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

        episode_size = image.size(0) // (self.way_num * (self.shot_num + self.query_num))

        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)


        acc_list = []

        for i in range(episode_size):
            s_label_onehot = self.labels_to_onehot(support_target[i])
            q_label_onehot = self.labels_to_onehot(query_target[i])
            
            _, acc = self.label_propagation(support_image[i], query_image[i], s_label_onehot, q_label_onehot)

            acc_list.append(acc)
        
        acc = torch.stack(acc_list)
        acc = torch.mean(acc) * 100.0

        return None, acc



