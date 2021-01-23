from abc import abstractmethod

import torch
from torch import nn

from core.utils import ModelType
from .init import init_weights


class AbstractModel(nn.Module):
    def __init__(self, way_num, shot_num, query_num, model_func, device, init_type,
                 model_type=ModelType.ABSTRACT):
        super(AbstractModel, self).__init__()
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.model_func = model_func
        self.device = device
        self.init_type = init_type
        self.model_type = model_type

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    def forward(self, x):
        out = self.model_func(x)
        return out

    def train(self, mode=True):
        return super(AbstractModel, self).train(mode)

    def eval(self):
        return super(AbstractModel, self).eval()

    def _init_network(self):
        init_weights(self, self.init_type)

    def _generate_local_targets(self, episode_size):
        local_targets = torch.arange(self.way_num, dtype=torch.long).view(1, -1, 1) \
            .repeat(episode_size, 1, self.shot_num + self.query_num).view(-1)
        return local_targets

    def split_by_episode(self, features, mode):
        """
        split features by episode and
        generate local targets + split labels by episode
        #!FIXME: 使用mode参数的方式需要调整，现在很难看
        """
        episode_size = features.size(0) // (self.way_num * (self.shot_num + self.query_num))
        local_labels = self._generate_local_targets(episode_size).to(self.device).contiguous().view(episode_size,
                                                                                                    self.way_num,
                                                                                                    self.shot_num + self.query_num)

        if mode == 1:  # input 2D, return 3D(with episode) E.g.ANIL & R2D2
            features = features.contiguous().view(episode_size, self.way_num, self.shot_num + self.query_num, -1)
            support_features = features[:, :, :self.shot_num, :].contiguous().view(episode_size,
                                                                                   self.way_num * self.shot_num, -1)
            query_features = features[:, :, self.shot_num:, :].contiguous().view(episode_size,
                                                                                 self.way_num * self.query_num, -1)
            support_targets = local_labels[:, :, :self.shot_num].contiguous().view(episode_size, -1)
            query_targets = local_labels[:, :, self.shot_num:].contiguous().view(-1)
        elif mode == 2:  # input 4D, return 5D(with episode) E.g.DN4
            b, c, h, w = features.shape
            features = features.contiguous().view(episode_size, self.way_num, self.shot_num + self.query_num, c, h, w)
            support_features = features[:, :, :self.shot_num, :, ...].contiguous().view(episode_size,
                                                                                        self.way_num * self.shot_num, c,
                                                                                        h, w)
            query_features = features[:, :, self.shot_num:, :, ...].contiguous().view(episode_size,
                                                                                      self.way_num * self.query_num, c,
                                                                                      h, w)
            support_targets = local_labels[:, :, :self.shot_num].contiguous().view(episode_size, -1)
            query_targets = local_labels[:, :, self.shot_num:].contiguous().view(-1)
        elif mode == 3:  # input 4D, return 4D(w/o episode) E.g.realationnet FIXME: 暂时用来处理还没有实现multi-task的方法
            b, c, h, w = features.shape
            features = features.contiguous().view(self.way_num, self.shot_num + self.query_num, c, h, w)
            support_features = features[:, :self.shot_num, :, ...].contiguous().view(self.way_num * self.shot_num, c,
                                                                                        h, w)
            query_features = features[ :, self.shot_num:, :, ...].contiguous().view(self.way_num * self.query_num, c,
                                                                                      h, w)
            support_targets = local_labels[:, :, :self.shot_num].contiguous().view(-1)
            query_targets = local_labels[:, :, self.shot_num:].contiguous().view(-1)
        elif mode ==4: #pretrain baseline input 2D, return 2D(w/o episode) E.g.baseline set_forward FIXME: 暂时用来处理还没有实现multi-task的方法
            features = features.view(self.way_num,self.shot_num + self.query_num,-1)
            support_features = features[:,:self.shot_num, :].contiguous().view(self.way_num * self.shot_num, -1)
            query_features = features[:,self.shot_num:, :].contiguous().view(self.way_num * self.query_num, -1)
            support_targets = local_labels[:, :, :self.shot_num].contiguous().view(-1)
            query_targets = local_labels[:, :, self.shot_num:].contiguous().view(-1)
        else:
            raise Exception("mode should in [1,2,3,4], not {}".format(mode))

        return support_features, query_features, support_targets, query_targets

    # def progress_batch(self, batch, ):
    #     images, _ = batch
    #     b, c, h, w = images.size()
    #     episode_size = b // (self.way_num * (self.shot_num + self.query_num))
    #     local_targets = self._generate_local_targets(episode_size)
    #
    #     assert episode_size == 1, \
    #         'only support for episode_size == 1, ' \
    #         'and current episode_size is {}'.format(episode_size)
    #
    #     images = images.view(self.way_num, self.shot_num + self.query_num, c, h, w)
    #     local_targets = local_targets.view(self.way_num, self.shot_num + self.query_num)
    #
    #     support_images = images[:, :self.shot_num, :, :, :].contiguous() \
    #         .view(self.way_num * self.shot_num, c, h, w).to(self.device)
    #     support_targets = local_targets[:, :self.shot_num].contiguous() \
    #         .view(self.way_num * self.shot_num).to(self.device)
    #     query_images = images[:, self.shot_num:, :, :, :].contiguous() \
    #         .view(self.way_num * self.query_num, c, h, w).to(self.device)
    #     query_targets = local_targets[:, self.shot_num:].contiguous() \
    #         .view(self.way_num * self.query_num).to(self.device)
    #
    #     return support_images, support_targets, query_images, query_targets
    #
    # def progress_batch2(self, batch, ):
    #     images, _ = batch
    #     b, c, h, w = images.size()
    #     episode_size = b // (self.way_num * (self.shot_num + self.query_num))
    #     local_targets = self._generate_local_targets(episode_size)
    #
    #     images = images.to(self.device)
    #     local_targets = local_targets.to(self.device)
    #
    #     return images, local_targets

    def reset_base_info(self, way_num, shot_num, query_num):
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
