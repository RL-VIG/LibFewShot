from .conv_64f import *
from .resnet_12 import *
from .resnet_18 import *


def get_backbone(config):
    kwargs = dict()
    kwargs.update(config['backbone']['kwargs'])
    if config['backbone']['name'] == 'Conv64F':
        model_func = Conv64F(**kwargs)
    elif config['backbone']['name'] == 'Conv64FLeakyReLU':
        model_func = Conv64FLeakyReLU(**kwargs)
    elif config['backbone']['name'] == 'ResNet12':
        model_func = resnet12(**kwargs)
    elif config['backbone']['name'] == 'ResNet18':
        model_func = resnet18(**kwargs)
    else:
        raise NotImplementedError

    return model_func
