from .collate_functions import *
from .contrib import get_augment_method
from ...utils import ModelType


def get_collate_function(config, trfms, mode, model_type, ):
    assert model_type != ModelType.ABSTRACT
    if mode == 'train' and model_type == ModelType.PRETRAIN:
        collate_function = GeneralCollateFunction(trfms, config['augment_times'])
    else:
        collate_function = FewShotAugCollateFunction(trfms, config['augment_times'],
                                                     config['train_way'] if mode == 'train'
                                         else config['test_way'],
                                                     config['train_shot'] if mode == 'train'
                                         else config['test_shot'],
                                                     config['train_query'] if mode == 'train'
                                         else config['test_query'],
                                                     config['episode_size'])

    return collate_function
