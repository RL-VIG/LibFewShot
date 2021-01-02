from .collate_fns import *

# TODO：differentiate fewshot_collate_fn & general_collate_fn


def get_general_collate_fn(config, trfms):
    try:
        collate_fn = GeneralCollateFn(trfms, config['augment_times'])
    except:
        raise NotImplementedError

    return collate_fn


def get_collate_fn(config, trfms):
    try:
        collate_fn = FewShotAugCollateFn(trfms, config['augment_times'])
    except:
        raise NotImplementedError

    return collate_fn
