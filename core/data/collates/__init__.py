from .collate_fns import *

from ...utils import ModelType


def get_collate_fn(config, trfms, mode, model_type, ):
    assert model_type != ModelType.ABSTRACT
    if mode == 'train' and model_type == ModelType.PRETRAIN:
        collate_fn = GeneralCollateFn(trfms, config['augment_times'])
    else:
        collate_fn = FewShotAugCollateFn(trfms, config['augment_times'])

    return collate_fn
