from abc import abstractmethod

from torch import nn

from core.model import init_weights
from core.utils import ModelType


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
