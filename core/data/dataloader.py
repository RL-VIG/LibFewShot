# -*- coding: utf-8 -*-
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from core.data.dataset import GeneralDataset
from .collates import get_collate_function, get_augment_method,get_mean_std
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
    assert (
        model_type != ModelType.ABSTRACT
    ), "model_type should not be ModelType.ABSTRACT"

    # Add user's trfms in get_augment_method()

    #get mean std
    MEAN,STD=get_mean_std(config, mode)
    
    trfms_list = get_augment_method(config, mode)

    trfms_list.append(transforms.ToTensor())
    trfms_list.append(transforms.Normalize(mean=MEAN, std=STD))
    trfms = transforms.Compose(trfms_list)

    dataset = GeneralDataset(
        data_root=config["data_root"],
        mode=mode,
        use_memory=config["use_memory"],
    )

    if config["dataloader_num"] == 1 or mode in ["val", "test"]:

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
            print(
                "with zero workers, the training phase will be very slow",
                level="warning",
            )

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

        return (dataloader,)
    else:
        # for RENet: use fs_loader and generic_loader in training stage
        collate_function = get_collate_function(config, trfms, mode, ModelType.METRIC)
        sampler = get_sampler(
            dataset=dataset,
            few_shot=True,
            distribute=distribute,
            mode=mode,
            config=config,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config["n_gpu"] * 4,
            pin_memory=True,
            collate_fn=collate_function,
        )
        collate_function = get_collate_function(
            config, trfms, mode, ModelType.FINETUNING
        )
        sampler = get_sampler(
            dataset=dataset,
            few_shot=False,
            distribute=distribute,
            mode=mode,
            config=config,
        )
        dataloader_aux = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["n_gpu"] * 4,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_function,
        )

        return (dataloader, dataloader_aux)


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
            object.__setattr__(
                self, "batch_sampler", _RepeatSampler(self.batch_sampler)
            )
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
