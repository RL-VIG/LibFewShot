# -*- coding: utf-8 -*-
from .autoaugment import ImageNetPolicy
from .cutout import Cutout
from .randaugment import RandAugment
from torchvision import transforms

CJ_DICT = {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4}


def get_augment_method(
    config,
):
    """Return the corresponding augmentation method according to the setting.

    + Use `ColorJitter` and `RandomHorizontalFlip` when not setting `augment_method` or using `NormalAug`.
    + Use `ImageNetPolicy()`when using `AutoAugment`.
    + Use `Cutout()`when using `Cutout`.
    + Use `RandAugment()`when using `RandAugment`.
    + Use `CenterCrop` and `RandomHorizontalFlip` when using `AutoAugment`.
    + Users can add their own augment method in this function.

    Args:
        config (dict): A LFS setting dict

    Returns:
        list: A list of specific transforms.
    """
    if "augment_method" not in config or config["augment_method"] == "NormalAug":
        trfms = [
            transforms.ColorJitter(**CJ_DICT),
            transforms.RandomHorizontalFlip(),
        ]
    elif config["augment_method"] == "AutoAugment":
        trfms = [ImageNetPolicy()]
    elif config["augment_method"] == "Cutout":
        trfms = [Cutout()]
    elif config["augment_method"] == "RandAugment":
        trfms = [RandAugment()]
    elif (
        config["augment_method"] == "MTLAugment"
    ):  # https://github.com/yaoyao-liu/meta-transfer-learning/blob/fe189c96797446b54a0ae1c908f8d92a6d3cb831/pytorch/dataloader/dataset_loader.py#L60
        trfms = [transforms.CenterCrop(80), transforms.RandomHorizontalFlip()]
    else:
        trfms = [
            transforms.ColorJitter(**CJ_DICT),
            transforms.RandomHorizontalFlip(),
        ]
    return trfms
