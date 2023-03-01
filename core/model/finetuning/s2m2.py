# -*- coding: utf-8 -*-

import pdb
from re import T
import torch
from torch import nn

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from torch.nn.utils.weight_norm import WeightNorm
import numpy as np
from torch.autograd import Variable
class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores
class S2M2(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(S2M2, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param

        self.classifier = nn.Linear(feat_dim, num_class)
        self.classifier_rot = nn.Linear(feat_dim, 4)
        self.disclass=distLinear(feat_dim, num_class)
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


        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat[i], support_target[i], query_feat[i])
            output_list.append(output)

        output = torch.cat(output_list, dim=0)


        acc = accuracy(output, query_target.reshape(-1))
        #acc=((output.to(self.device)==query_target).sum().item()/output.size(0))
        return output, acc

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        #use manifold-mixup
        new_chunks = []
        sizes = torch.chunk(target, 1)

        for i in range(1):
            new_chunks.append(torch.randperm(sizes[i].shape[0]))
        index_mixup = torch.cat(new_chunks, dim = 0)
        lam = np.random.beta(2, 2)

        feat = self.emb_func(image, index_mixup = index_mixup, lam = lam)
        output=self.disclass(feat)
        loss_mm=lam*self.loss_func(output, target)+(1-lam)*self.loss_func(output, target[index_mixup])
        acc = accuracy(output, target)

        #use rotation
        image_rot,target_class,target_rot_class=self.rot_image_generation(image,target)
        feat = self.emb_func(image_rot)

        output_class=self.disclass(feat)
        output_rot_class=self.classifier_rot(feat)
        loss=0.5*self.loss_func(output_class, target_class) +0.5*self.loss_func(output_rot_class, target_rot_class)

        loss_re=loss+loss_mm
        acc = accuracy(output_class, target_class)
        return output, acc, loss_re

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        classifier = distLinear(self.feat_dim, self.test_way)
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)

        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                optimizer.zero_grad()
                select_id = rand_id[i : min(i + self.inner_param["inner_batch_size"], support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]
                #print(batch.size())
                output = classifier(batch)

                loss = self.loss_func(output, target)


                loss.backward()
                optimizer.step()

        output = classifier(query_feat)
        return output
    
    def  rot_image_generation(self, image, target): 
        bs=image.shape[0]
        indices = np.arange(bs)
        np.random.shuffle(indices)
        split_size=bs//4
        image_rot=[]
        target_class =[]
        target_rot_class=[]

        for j in indices[0:split_size]:
            x90 = image[j].transpose(2,1).flip(1)
            x180 = x90.transpose(2,1).flip(1)
            x270 =  x180.transpose(2,1).flip(1)
            image_rot += [image[j], x90, x180, x270]
            target_class += [target[j] for _ in range(4)]
            target_rot_class += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]
        image_rot =torch.stack(image_rot,0).to(self.device)
        target_class =torch.stack(target_class,0).to(self.device)
        target_rot_class=torch.stack(target_rot_class,0).to(self.device)
        image_rot=torch.tensor(image_rot).to(self.device)
        target_class=torch.tensor(target_class).to(self.device)
        target_rot_class=torch.tensor(target_rot_class).to(self.device)
        return image_rot,target_class,target_rot_class