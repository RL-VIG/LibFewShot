from .collate_fns import *
from .contrib import get_augment_method
from ...utils import ModelType


def get_collate_fn(config, trfms, mode, model_type, ):
    assert model_type != ModelType.ABSTRACT
    if mode == 'train' and model_type == ModelType.PRETRAIN:
        collate_fn = GeneralCollateFn(trfms, config['augment_times'])
    else:
        collate_fn = FewShotAugCollateFn(trfms, config['augment_times'],config['augment_times_query'],
                                         config['way_num'] if mode == 'train'
                                         else config['test_way'],
                                         config['shot_num'] if mode == 'train'
                                         else config['test_shot'],
                                         config['query_num'] if mode == 'train'
                                         else config['test_query'],
                                         config['episode_size'])

    return collate_fn
