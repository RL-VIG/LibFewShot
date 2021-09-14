# -*- coding: utf-8 -*-
from enum import Enum


class ModelType(Enum):
    ABSTRACT = 0
    FINETUNING = 1
    METRIC = 2
    META = 3


class SaveType(Enum):
    NORMAL = 0
    BEST = 1
    LAST = 2
