# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms

from core.data.dataset import GeneralDataset
from .collates import get_collate_function, get_augment_method
from .samplers import CategoriesSampler
from ..utils import ModelType

MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]


def get_dataloader(config, mode, model_type):
    """Get the dataloader corresponding to the model type and training phase.

    According to the config dict, the training phase and model category, select the appropriate transforms, set the corresponding sampler and collate_fn, and return the corresponding dataloader.

    Args:
        config (dict): A LibFewShot setting dict
        mode (str): mode in train/test/val
        model_type (ModelType): model type in meta/metric//finetuning

    Returns:
        Dataloader: The corresponding dataloader.
    """
    assert model_type != ModelType.ABSTRACT

    trfms_list = []

    # Add user's trfms here (or in get_augment_method())
    if mode == "train" and config["augment"]:
        if config["image_size"] == 224:
            trfms_list.append(transforms.Resize((256, 256)))
            trfms_list.append(transforms.RandomCrop((224, 224)))
        elif config["image_size"] == 84:
            trfms_list.append(transforms.Resize((96, 96)))
            trfms_list.append(transforms.RandomCrop((84, 84)))
        # for MTL -> alternative solution: use avgpool(ks=11)
        elif config["image_size"] == 80:
            # MTL use another MEAN and STD
            trfms_list.append(transforms.Resize((92, 92)))
            trfms_list.append(transforms.RandomResizedCrop(88))
            trfms_list.append(transforms.CenterCrop((80, 80)))
            trfms_list.append(transforms.RandomHorizontalFlip())
        else:
            raise RuntimeError

        aug_method = get_augment_method(config)
        trfms_list += aug_method
    else:
        if config["image_size"] == 224:
            trfms_list.append(transforms.Resize((256, 256)))
            trfms_list.append(transforms.CenterCrop((224, 224)))
        elif config["image_size"] == 84:
            trfms_list.append(transforms.Resize((96, 96)))
            trfms_list.append(transforms.CenterCrop((84, 84)))
        # for MTL -> alternative solution: use avgpool(ks=11)
        elif config["image_size"] == 80:
            trfms_list.append(transforms.Resize((92, 92)))
            trfms_list.append(transforms.CenterCrop((80, 80)))
        else:
            raise RuntimeError

    trfms_list.append(transforms.ToTensor())
    trfms_list.append(transforms.Normalize(mean=MEAN, std=STD))
    trfms = transforms.Compose(trfms_list)

    dataset = GeneralDataset(
        data_root=config["data_root"],
        mode=mode,
        use_memory=config["use_memory"],
    )

    collate_function = get_collate_function(config, trfms, mode, model_type)

    if mode == "train" and model_type == ModelType.FINETUNING:
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["n_gpu"] * 4,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_function,
        )
    else:
        sampler = CategoriesSampler(
            label_list=dataset.label_list,
            label_num=dataset.label_num,
            episode_size=config["episode_size"],
            episode_num=config["train_episode"] if mode == "train" else config["test_episode"],
            way_num=config["way_num"] if mode == "train" else config["test_way"],
            image_num=config["shot_num"] + config["query_num"]
            if mode == "train"
            else config["test_shot"] + config["test_query"],
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config["n_gpu"] * 4,
            pin_memory=True,
            collate_fn=collate_function,
        )

    return dataloader
