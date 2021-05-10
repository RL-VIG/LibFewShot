from .autoaugment import ImageNetPolicy
from .cutout import Cutout
from .randaugment import RandAugment
from torchvision import transforms

CJ_DICT = {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4}


def get_augment_method(config,):
    """
    根据配置文件返回相应的增广方式

    当为指定增广方法或使用`NormalAug`时，使用`ColorJitter`和`RandomHorizontalFlip，····（其他）

    Args:
        config (dict): A LFS setting dict

    Returns:
        list: a list of specific transforms.
    """
    if "augment_method" not in config or config["augment_method"] == "NormalAug":
        trfms = [transforms.ColorJitter(
            **CJ_DICT), transforms.RandomHorizontalFlip()]
    elif config["augment_method"] == "AutoAugment":
        trfms = [ImageNetPolicy()]
    elif config["augment_method"] == "Cutout":
        trfms = [Cutout()]
    elif config["augment_method"] == "RandAugment":
        trfms = [RandAugment()]
    elif config["augment_method"] == "MTLAugment":
        trfms = [transforms.CenterCrop(80), transforms.RandomHorizontalFlip()]

    return trfms
