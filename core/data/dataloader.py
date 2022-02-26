# -*- coding: utf-8 -*-
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from core.data.dataset import GeneralDataset
from .collates import get_collate_function, get_augment_method
from .samplers import DistributedCategoriesSampler, get_sampler
from ..utils import ModelType

MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

import torch
from queue import Queue
from threading import Thread


def get_dataloader(config, mode, model_type, distribute):
    """Get the dataloader corresponding to the model type and training phase.

    According to the config dict, the training phase and model category, select the appropriate transforms, set the corresponding sampler and collate_fn, and return the corresponding dataloader.

    Args:
        config (dict): A LibFewShot setting dict
        mode (str): mode in train/test/val
        model_type (ModelType): model type in meta/metric//finetuning

    Returns:
        Dataloader: The corresponding dataloader.
    """
    assert model_type != ModelType.ABSTRACT, "model_type should not be ModelType.ABSTRACT"

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
    assert dataset.label_num >= (
        config["way_num"] if mode == "train" else config["test_way"]
    ), "classes({}) in {} split should be larger than {}({})".format(
        dataset.label_num,
        mode,
        "way_num" if mode == "train" else "test_way",
        (config["way_num"] if mode == "train" else config["test_way"]),
    )

    # for mixup or other augmentations
    if config["aug_config"] and mode in config["aug_config"] and config["aug_config"][mode] and "mixup" in config["aug_config"][mode]:
        config["aug_config"][mode]["mixup"]["num_classes"] = dataset.label_num
    collate_function = get_collate_function(config, trfms, mode, model_type)

    few_shot = not (model_type == ModelType.FINETUNING and mode == "train")

    sampler = get_sampler(
        dataset=dataset,
        few_shot=few_shot,
        distribute=distribute,
        mode=mode,
        config=config,
    )

    data_scale = 1 if config["n_gpu"] == 0 else config["n_gpu"]
    workers = config["workers"] // data_scale
    if workers == 0:
        print("with zero workers, the training phase will be very slow", level="warning")

    dataloader = MultiEpochsDataLoader(
        dataset=dataset,
        sampler=None if few_shot else sampler,
        batch_sampler=sampler if few_shot else None,
        batch_size=1
        if few_shot
        else (config["batch_size"] // data_scale),  # batch_size is default set to 1
        shuffle=False if few_shot or distribute else True,
        num_workers=workers,  # num_workers for each gpu
        drop_last=False if few_shot else True,
        pin_memory=True,
        collate_fn=collate_function,
    )

    return dataloader


# https://www.zhihu.com/question/307282137/answer/1560137140
class _RepeatSampler(object):
    """repeated sampler"""

    def __init__(self, sampler):
        self.sampler = sampler
        self.repeat_sample = True if len(self.sampler) > 0 else False

    def __iter__(self):
        while self.repeat_sample:
            yield from iter(self.sampler)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)


class MultiEpochsDataLoader(DataLoader):
    """
    When training with multiple epochs,
    the DataLoader object does not have to re-create the thread and the batch_sampler object to save initialization time for each epoch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.few_shot = False
        if self.batch_sampler is not None:
            object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
            self.iterator = super().__iter__()
            self.few_shot = True

    def __len__(self):
        if self.few_shot:
            return len(self.batch_sampler.sampler)
        else:
            super().__len__()

    def __iter__(self):
        if self.few_shot:
            for i in range(len(self)):
                yield next(self.iterator)
        else:
            super().__iter__()
