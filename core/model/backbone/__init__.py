from .conv_64f import *


def get_backbone(config):
    kwargs = dict()
    kwargs.update(config['backbone']['kwargs'])
    if config['backbone']['name'] == 'Conv64F':
        model_func = Conv64F(**kwargs)
    elif config['backbone']['name'] == 'Conv64FLeakyReLU':
        model_func = Conv64FLeakyReLU(**kwargs)
    else:
        raise NotImplementedError

    return model_func
