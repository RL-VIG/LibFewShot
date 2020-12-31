from abc import abstractmethod

from core.model.abstract_model import AbstractModel
from core.utils import ModelType


class MetricModel(AbstractModel):

    def __init__(self, way_num, shot_num, query_num, model_func, device,
                 init_type='normal'):
        super(MetricModel, self).__init__(way_num, shot_num, query_num, model_func,
                                          device, init_type, ModelType.METRIC)

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    def forward(self, x):
        out = self.model_func(x)
        return out
