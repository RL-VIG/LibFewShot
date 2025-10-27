# -*- coding: utf-8 -*-
from .conv_four import Conv32F, Conv64F, R2D2Embedding
from .conv_four_mcl import Conv64F_MCL
from .resnet_12 import resnet12, resnet12woLSC
from .resnet_12_mcl import resnet12_mcl, resnet12_r2d2
from .resnet_18 import resnet18
from .wrn import WRN
from .resnet_12_mtl_offcial import resnet12MTLofficial
from .vit import ViT
from .swin_transformer import swin_s, swin_l, swin_b, swin_t, swin_mini
from .resnet_bdc import resnet12Bdc, resnet18Bdc
from .fgfl_resnet12 import fgfl_resnet12  # FGFL-specific ResNet-12
from core.model.backbone.utils.maml_module import convert_maml_module


def get_backbone(config):
    """Get the backbone according to the config dict.

    Args:
        config: The config dict.

    Returns: The backbone module.

    """
    kwargs = dict()
    kwargs.update(config["backbone"]["kwargs"])
    try:
        emb_func = eval(config["backbone"]["name"])(**kwargs)
    except NameError:
        raise NotImplementedError(
            "{} is not implemented".format(config["backbone"]["name"])
        )

    return emb_func
