from abc import abstractmethod

import torch
from torch import nn

from core.utils import ModelType
from .init import init_weights


class AbstractModel(nn.Module):
    def __init__(self, train_way, train_shot, train_query, emb_func, device, init_type,
                 model_type=ModelType.ABSTRACT):
        super(AbstractModel, self).__init__()
        self.train_way = train_way
        self.train_shot = train_shot
        self.train_query = train_query
        self.emb_func = emb_func
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
        out = self.emb_func(x)
        return out

    def train(self, mode=True):
        super(AbstractModel, self).train(mode)
        # for methods with distiller
        if hasattr(self, 'distill_layer'):
            self.distill_layer.train(False)

    def eval(self):
        return super(AbstractModel, self).eval()

    def _init_network(self):
        init_weights(self, self.init_type)

    def _generate_local_targets(self, episode_size):
        local_targets = torch.arange(self.train_way, dtype=torch.long).view(1, -1, 1) \
            .repeat(episode_size, 1, self.train_shot + self.train_query).view(-1)
        return local_targets

    def split_by_episode(self, features, mode):
        """
        split features by episode and
        generate local targets + split labels by episode
        #!FIXME: 使用mode参数的方式需要调整，现在很难看
        """
        episode_size = features.size(0) // (self.train_way * (self.train_shot + self.train_query))
        local_labels = self._generate_local_targets(episode_size).to(self.device).contiguous().view(episode_size,
                                                                                                    self.train_way,
                                                                                                    self.train_shot +
                                                                                                    self.train_query)

        if mode == 1:  # input 2D, return 3D(with episode) E.g.ANIL & R2D2
            features = features.contiguous().view(episode_size, self.train_way, self.train_shot + self.train_query, -1)
            support_features = features[:, :, :self.train_shot, :].contiguous().view(episode_size,
                                                                                   self.train_way * self.train_shot, -1)
            query_features = features[:, :, self.train_shot:, :].contiguous().view(episode_size,
                                                                                 self.train_way * self.train_query, -1)
            support_target = local_labels[:, :, :self.train_shot].contiguous().view(episode_size, -1)
            query_target = local_labels[:, :, self.train_shot:].contiguous().view(episode_size, -1)
        elif mode == 2:  # input 4D, return 5D(with episode) E.g.DN4
            b, c, h, w = features.shape
            features = features.contiguous().view(episode_size, self.train_way, self.train_shot + self.train_query, c, h, w)
            support_features = features[:, :, :self.train_shot, :, ...].contiguous().view(episode_size,
                                                                                        self.train_way * self.train_shot, c,
                                                                                        h, w)
            query_features = features[:, :, self.train_shot:, :, ...].contiguous().view(episode_size,
                                                                                      self.train_way * self.train_query, c,
                                                                                      h, w)
            support_target = local_labels[:, :, :self.train_shot].contiguous().view(episode_size, -1)
            query_target = local_labels[:, :, self.train_shot:].contiguous().view(-1)
        elif mode == 3:  # input 4D, return 4D(w/o episode) E.g.realationnet FIXME: 暂时用来处理还没有实现multi-task的方法
            b, c, h, w = features.shape
            features = features.contiguous().view(self.train_way, self.train_shot + self.train_query, c, h, w)
            support_features = features[:, :self.train_shot, :, ...].contiguous().view(self.train_way * self.train_shot, c,
                                                                                     h, w)
            query_features = features[:, self.train_shot:, :, ...].contiguous().view(self.train_way * self.train_query, c,
                                                                                   h, w)
            support_targets = local_labels[:, :, :self.train_shot].contiguous().view(-1)
            query_targets = local_labels[:, :, self.train_shot:].contiguous().view(-1)
        elif mode == 4:  # pretrain baseline input 2D, return 2D(w/o episode) E.g.baseline set_forward FIXME:
            # 暂时用来处理还没有实现multi-task的方法
            features = features.view(self.train_way, self.train_shot + self.train_query, -1)
            support_features = features[:, :self.train_shot, :].contiguous().view(self.train_way * self.train_shot, -1)
            query_features = features[:, self.train_shot:, :].contiguous().view(self.train_way * self.train_query, -1)
            support_target = local_labels[:, :, :self.train_shot].contiguous().view(-1)
            query_target = local_labels[:, :, self.train_shot:].contiguous().view(-1)
        else:
            raise Exception("mode should in [1,2,3,4], not {}".format(mode))

        return support_features, query_features, support_target, query_target

    # def progress_batch(self, batch, ):
    #     images, _ = batch
    #     b, c, h, w = images.size()
    #     episode_size = b // (self.train_way * (self.train_shot + self.train_query))
    #     local_targets = self._generate_local_targets(episode_size)
    #
    #     assert episode_size == 1, \
    #         'only support for episode_size == 1, ' \
    #         'and current episode_size is {}'.format(episode_size)
    #
    #     images = images.view(self.train_way, self.train_shot + self.train_query, c, h, w)
    #     local_targets = local_targets.view(self.train_way, self.train_shot + self.train_query)
    #
    #     support_images = images[:, :self.train_shot, :, :, :].contiguous() \
    #         .view(self.train_way * self.train_shot, c, h, w).to(self.device)
    #     support_target = local_targets[:, :self.train_shot].contiguous() \
    #         .view(self.train_way * self.train_shot).to(self.device)
    #     query_images = images[:, self.train_shot:, :, :, :].contiguous() \
    #         .view(self.train_way * self.train_query, c, h, w).to(self.device)
    #     query_target = local_targets[:, self.train_shot:].contiguous() \
    #         .view(self.train_way * self.train_query).to(self.device)
    #
    #     return support_images, support_target, query_images, query_target
    #
    # def progress_batch2(self, batch, ):
    #     images, _ = batch
    #     b, c, h, w = images.size()
    #     episode_size = b // (self.train_way * (self.train_shot + self.train_query))
    #     local_targets = self._generate_local_targets(episode_size)
    #
    #     images = images.to(self.device)
    #     local_targets = local_targets.to(self.device)
    #
    #     return images, local_targets

    def reset_base_info(self, train_way, train_shot, train_query):
        self.train_way = train_way
        self.train_shot = train_shot
        self.train_query = train_query
