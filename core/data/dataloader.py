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


# https://www.zhihu.com/question/307282137/answer/1560137140
class CudaDataLoader:
    """
    Asynchronously preloads data from the CPU to the GPU
    """

    def __init__(self, loader, device, queue_size=2):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        # The loop that will load into the queue in the background
        torch.cuda.set_device(self.device)
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) == str:
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # process is dead
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # end a epoch
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # next batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class _RepeatSampler(object):
    """repeated sampler"""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
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
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)



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

    # dataset = torchvision.datasets.ImageNet("/data/IMAGENET2012/", split="train")

    # dataset = GeneralDataset("/data/yxs/", train=True, download=False)

    dataset = GeneralDataset(
        data_root=config["data_root"],
        mode=mode,
        use_memory=config["use_memory"],
    )

    collate_function = get_collate_function(config, trfms, mode, model_type)

    few_shot = (model_type != ModelType.FINETUNING)

    sampler = get_sampler(
        dataset=dataset,
        few_shot = few_shot,
        distribute=distribute,
        mode=mode,
        config=config
    )

    dataloader = MultiEpochsDataLoader(
        dataset=dataset,
        sampler=None if few_shot else sampler,
        batch_sampler=sampler if few_shot else None,
        batch_size= 1 if few_shot else config["batch_size"],    # batch_size is default set to 1
        shuffle=False if few_shot and distribute else False,
        num_workers=4,                                          # num_workers for each gpu
        drop_last=False if few_shot else True,
        pin_memory=True,
        collate_fn=collate_function,
    )

    # if torch.cuda.is_available():
    #     dataloader = CudaDataLoader(dataloader, config["rank"])

    return dataloader
