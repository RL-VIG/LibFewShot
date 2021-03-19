from .collate_fns import *
from .contrib import get_augment_method
from ...utils import ModelType


def get_collate_fn(config, trfms, mode, model_type, ):
    assert model_type != ModelType.ABSTRACT
    if mode == 'train' and model_type == ModelType.PRETRAIN:
        collate_fn = GeneralCollateFn(trfms, config['augment_times'])
    else:
        collate_fn = FewShotAugCollateFn(trfms, config['augment_times'], config['way_num'],config['shot_num'],config['query_num'],config['episode_size'])

    return collate_fn
