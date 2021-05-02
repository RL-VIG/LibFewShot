from abc import abstractmethod

import torch

from core.model.abstract_model import AbstractModel
from core.utils import ModelType


class PretrainModel(AbstractModel):
    def __init__(
        self, way_num, shot_num, query_num, emb_func, device, init_type="normal"
    ):
        super(PretrainModel, self).__init__(
            way_num,
            shot_num,
            query_num,
            emb_func,
            device,
            init_type,
            ModelType.PRETRAIN,
        )

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    def forward(self, x):
        out = self.emb_func(x)
        return out

    @abstractmethod
    def test_loop(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_adaptation(self, *args, **kwargs):
        pass

    def sub_optimizer(self, model, config):
        kwargs = dict()

        if config["kwargs"] is not None:
            kwargs.update(config["kwargs"])
        return getattr(torch.optim, config["name"])(model.parameters(), **kwargs)
