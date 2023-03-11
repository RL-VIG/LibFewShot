# -*- coding: utf-8 -*-
from .autoaugment import ImageNetPolicy
from .cutout import Cutout
from .randaugment import RandAugment
from torchvision import transforms

CJ_DICT = {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4}


def get_augment_method(
    config,
    mode,
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
        mode (str): mode in train/test/val

    Returns:
        list: A list of specific transforms.
    """
    if mode == "train" and config["augment"]:
        # Add user's trfms here
        if "augment_method" not in config or config["augment_method"] == "NormalAug":
            trfms_list = get_default_image_size_trfms(config["image_size"])
            trfms_list += [
                transforms.ColorJitter(**CJ_DICT),
                transforms.RandomHorizontalFlip(),
            ]
        elif config["augment_method"] == "AutoAugment":
            trfms_list = get_default_image_size_trfms(config["image_size"])
            trfms_list += [ImageNetPolicy()]
        elif config["augment_method"] == "Cutout":
            trfms_list = get_default_image_size_trfms(config["image_size"])
            trfms_list += [Cutout()]
        elif config["augment_method"] == "RandAugment":
            trfms_list = get_default_image_size_trfms(config["image_size"])
            trfms_list += [RandAugment()]
        elif config["augment_method"] == "MTLAugment":  
            trfms_list = get_default_image_size_trfms(config["image_size"])
            # https://github.com/yaoyao-liu/meta-transfer-learning/blob/fe189c96797446b54a0ae1c908f8d92a6d3cb831/pytorch/dataloader/dataset_loader.py#L60
            trfms_list += [transforms.CenterCrop(80), transforms.RandomHorizontalFlip()]
        elif config["augment_method"] == "DeepBdcAugment":
            # https://github.com/Fei-Long121/DeepBDC/blob/main/data/datamgr.py#23
            trfms_list = [
                transforms.RandomResizedCrop(config["image_size"]),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(**CJ_DICT),
            ]
        elif config["augment_method"] == "S2M2Augment":
            trfms_list = [
                transforms.RandomResizedCrop(config["image_size"]),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(**CJ_DICT),
            ]
        else:
            trfms_list = get_default_image_size_trfms(config["image_size"])
            trfms_list += [
                transforms.ColorJitter(**CJ_DICT),
                transforms.RandomHorizontalFlip(),
            ]
            
    else:
        if config["image_size"] == 224:
            trfms_list = [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
            ]
        elif config["image_size"] == 84:
            trfms_list = [
                transforms.Resize((96, 96)),
                transforms.CenterCrop((84, 84)),
            ]
        # for MTL -> alternative solution: use avgpool(ks=11)
        elif config["image_size"] == 80:
            trfms_list = [
                transforms.Resize((92, 92)),
                transforms.CenterCrop((80, 80)),
            ]
        else:
            raise RuntimeError

    return trfms_list


def get_default_image_size_trfms(image_size):
    """ Return the uniform transforms for image_size """
    if image_size == 224:
        trfms = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
        ]
    elif image_size == 84:
        trfms = [
            transforms.Resize((96, 96)),
            transforms.RandomCrop((84, 84)),
        ]
    # for MTL -> alternative solution: use avgpool(ks=11)
    elif image_size == 80:
        # MTL use another MEAN and STD
        trfms = [
            transforms.Resize((92, 92)),
            transforms.RandomResizedCrop(88),
            transforms.CenterCrop((80, 80)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        raise RuntimeError
    return trfms

def get_mean_std(
    config,
    mode,
):
    """Return the corresponding mean and std according to the setting.


    Args:
        config (dict): A LFS setting dict
        mode (str): mode in train/test/val

    Returns:
        list: A list of specific transforms.
        
    """

    MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
    STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
    
    if "augment_method" not in config or config["augment_method"] == "NormalAug":
        MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
    elif config["augment_method"]== "S2M2Augment":
        MEAN= [0.485, 0.456, 0.406]
        STD=[0.229, 0.224, 0.225]
    return MEAN,STD