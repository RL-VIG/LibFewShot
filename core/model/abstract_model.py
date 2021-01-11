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

    def progress_batch(self, batch, ):
        images, _ = batch
        b, c, h, w = images.size()
        episode_size = b // (self.way_num * (self.shot_num + self.query_num))
        local_targets = self._generate_local_targets(episode_size)

        assert episode_size == 1, \
            'only support for episode_size == 1, ' \
            'and current episode_size is {}'.format(episode_size)

        images = images.view(self.way_num, self.shot_num + self.query_num, c, h, w)
        local_targets = local_targets.view(self.way_num, self.shot_num + self.query_num)

        support_images = images[:, :self.shot_num, :, :, :].contiguous() \
            .view(self.way_num * self.shot_num, c, h, w).to(self.device)
        support_targets = local_targets[:, :self.shot_num].contiguous() \
            .view(self.way_num * self.shot_num).to(self.device)
        query_images = images[:, self.shot_num:, :, :, :].contiguous() \
            .view(self.way_num * self.query_num, c, h, w).to(self.device)
        query_targets = local_targets[:, self.shot_num:].contiguous() \
            .view(self.way_num * self.query_num).to(self.device)

        return support_images, support_targets, query_images, query_targets

    def progress_batch2(self, batch, ):
        images, _ = batch
        b, c, h, w = images.size()
        episode_size = b // (self.way_num * (self.shot_num + self.query_num))
        local_targets = self._generate_local_targets(episode_size)

        images = images.to(self.device)
        local_targets = local_targets.to(self.device)

        return images, local_targets

    def reset_base_info(self, way_num, shot_num, query_num):
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
