from .collate_fns import *

method_dict = {
    'sample5': SampleFiveTimesAugCollateFn,  # can't access, replace it
}


def get_collate_fn(config, trfms):
    kwargs = {
        'trfms': trfms,
    }
    if not config['augment']:
        collate_fn = VanillaCollateFn(**kwargs)
    else:
        try:
            collate_fn = method_dict[config['augment_method']](**kwargs)
        except:
            raise NotImplementedError

    return collate_fn
