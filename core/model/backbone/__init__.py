from .conv_64f import Conv64F, Conv64FReLU, Conv64FLeakyReLU
from .conv_32f import Conv32F, Conv32FReLU, Conv32FLeakyReLU
from .resnet_12 import resnet12
from .resnet_18 import resnet18
from .maml_backbone import Conv32F_fw, Conv64F_fw
from .conv_64five import Conv64Five
from .wrn import WRN
from .resnet_12_mtl_original import resnet12MTL
from .resnet_12_mtl import resnet12mtlori
from .resnet_18_mtl import resnet18mtlori
from .conv_64f_mtl import Conv64FMTL

def get_backbone(config):
    kwargs = dict()
    kwargs.update(config['backbone']['kwargs'])
    try:
        model_func = eval(config['backbone']['name'])(**kwargs)
    except NameError:
        raise("{} is not implemented".format(config['backbone']['name']))

    return model_func
