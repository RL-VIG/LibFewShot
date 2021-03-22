from .conv_64f import *
from .conv_32f import *
from .resnet_12 import *
from .resnet_18 import *
from .maml_backbone import *
from .conv_64five import *
from .wrn import *
from  .resnet_12_mtl import *

def get_backbone(config):
    kwargs = dict()
    kwargs.update(config['backbone']['kwargs'])
    if config['backbone']['name'] == 'Conv64F':
        model_func = Conv64F(**kwargs)
    elif config['backbone']['name'] == 'Conv32F':
        model_func = Conv32F(**kwargs)
    elif config['backbone']['name'] == 'Conv64FLeakyReLU':
        model_func = Conv64FLeakyReLU(**kwargs)
    elif config['backbone']['name'] == 'Conv64FReLU':
        model_func = Conv64FReLU(**kwargs)
    elif config['backbone']['name'] == 'ResNet12':
        model_func = resnet12(**kwargs)
    elif config['backbone']['name'] == 'ResNet18':
        model_func = resnet18(**kwargs)
    elif config['backbone']['name'] == 'ResNet12MTL':
        model_func = resnet12MTL()(**kwargs)
    elif config['backbone']['name'] == 'Conv64F_fw':
        model_func = Conv64F_fw(**kwargs)
    elif config['backbone']['name'] == 'Conv32F_fw':
        model_func = Conv32F_fw(**kwargs)
    elif config['backbone']['name'] == 'Conv64Five':
        model_func = Conv64Five(**kwargs)
    elif config['backbone']['name'] == 'WRN':
        model_func = WRN(**kwargs)
    else:
        raise NotImplementedError

    return model_func
