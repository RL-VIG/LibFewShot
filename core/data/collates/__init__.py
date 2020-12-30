from .collate_fns import *


def get_collate_fn(config, trfms):
    try:
        collate_fn = FewShotAugCollateFn(trfms, config['augment_times'])
    except:
        raise NotImplementedError

    return collate_fn
