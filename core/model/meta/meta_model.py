# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch

from core.model.abstract_model import AbstractModel
from core.utils import ModelType


class MetaModel(AbstractModel):
    def __init__(self, init_type="normal", **kwargs):
        super(MetaModel, self).__init__(init_type, ModelType.META, **kwargs)

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_adaptation(self, *args, **kwargs):
        pass

    def sub_optimizer(self, parameters, config):
        kwargs = dict()

        if config["kwargs"] is not None:
            kwargs.update(config["kwargs"])
        return getattr(torch.optim, config["name"])(parameters, **kwargs)
