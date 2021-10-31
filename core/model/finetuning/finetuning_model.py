# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch

from core.model.abstract_model import AbstractModel
from core.utils import ModelType


class FinetuningModel(AbstractModel):
    def __init__(self, init_type="normal", **kwargs):
        super(FinetuningModel, self).__init__(init_type, ModelType.FINETUNING, **kwargs)

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_adaptation(self, *args, **kwargs):
        pass

    def sub_optimizer(self, model, config):
        kwargs = dict()

        if config["kwargs"] is not None:
            kwargs.update(config["kwargs"])
        return getattr(torch.optim, config["name"])(model.parameters(), **kwargs)
