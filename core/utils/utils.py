# -*- coding: utf-8 -*-
import errno
import os
import random
from collections import OrderedDict
from datetime import datetime
from logging import getLogger

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import torch
import torch.multiprocessing

from core.utils import SaveType


def get_instance(module, name, config, **kwargs):
    """
    A reflection function to get backbone/classifier/.

    Args:
        module ([type]): Package Name.
        name (str): Top level value in config dict. (backbone, classifier, etc.)
        config (dict): The parsed config dict.

    Returns:
         Corresponding instance.
    """
    if config[name]["kwargs"] is not None:
        kwargs.update(config[name]["kwargs"])

    return getattr(module, config[name]["name"])(**kwargs)


class AverageMeter(object):
    """
    A AverageMeter to meter avg of number-like data.
    """

    def __init__(self, name, keys, writer=None):
        self.name = name
        self._data = pd.DataFrame(index=keys, columns=["last_value", "total", "counts", "average"])
        self.writer = writer
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            tag = "{}/{}".format(self.name, key)
            self.writer.add_scalar(tag, value)
        self._data.last_value[key] = value
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def last(self, key):
        return self._data.last_value[key]


def get_local_time():
    cur_time = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")

    return cur_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(output, target, topk=1):
    """
    Calc the acc of tpok.

    output and target have the same dtype and the same shape.

    Args:
        output (torch.Tensor or np.ndarray): The output.
        target (torch.Tensor or np.ndarray): The target.
        topk (int or list or tuple): topk . Defaults to 1.

    Returns:
        float: acc.
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = {
            "Tensor": torch.topk,
            "ndarray": lambda output, maxk, axis: (
                None,
                torch.from_numpy(topk_(output, maxk, axis)[1]).to(target.device),
            ),
        }[output.__class__.__name__](output, topk, 1)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size).item()
        return res


def topk_(matrix, K, axis):
    """
    the function to calc topk acc of ndarrary.

    TODO

    """
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort


def mean_confidence_interval(data, confidence=0.95):
    """

    :param data:
    :param confidence:
    :return:
    """
    a = [1.0 * np.array(data[i]) for i in range(len(data))]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def create_dirs(dir_paths):
    """

    :param dir_paths:
    :return:
    """
    if not isinstance(dir_paths, (list, tuple)):
        dir_paths = [dir_paths]
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


def prepare_device(device_ids, n_gpu_use):
    """

    :param n_gpu_use:
    :return:
    """
    logger = getLogger(__name__)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids)

    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("the model will be performed on CPU.")
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        logger.warning(
            "only {} are available on this machine, "
            "but the number of the GPU in config is {}.".format(n_gpu, n_gpu_use)
        )
        n_gpu_use = n_gpu

    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))

    return device, list_ids


def save_model(
    model,
    optimizer,
    lr_Scheduler,
    save_path,
    name,
    epoch,
    save_type=SaveType.LAST,
    is_parallel=False,
):
    """

    :param model:
    :param optimizer:
    :param lr_Scheduler
    :param save_path:
    :param name:
    :param epoch:
    :param save_type:
    :param is_parallel:
    :return:
    """

    if save_type == SaveType.NORMAL:
        save_name = os.path.join(save_path, "{}_{:0>5d}.pth".format(name, epoch))
    elif save_type == SaveType.BEST:
        save_name = os.path.join(save_path, "{}_best.pth".format(name))
    elif save_type == SaveType.LAST:
        save_name = os.path.join(save_path, "{}_last.pth".format(name))

    else:
        raise RuntimeError

    if is_parallel:
        model_state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            name = ".".join([name for name in k.split(".") if name != "module"])
            model_state_dict[name] = v
    else:
        model_state_dict = model.state_dict()

    if save_type == SaveType.NORMAL or save_type == SaveType.BEST:
        torch.save(model_state_dict, save_name)
    else:
        torch.save(
            {
                "epoch": epoch,
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_Scheduler.state_dict(),
            },
            save_name,
        )

    return save_name


def init_seed(seed=0, deterministic=False):
    """

    :param seed:
    :param deterministic:
    :return:
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
class data_prefetcher:
    """
    make dataloader more fast.
    """

    def __init__(self, loader):
        """
        loader: train_loader, val_loader or test_loader
        """
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            self.next_data = [data.cuda(non_blocking=True) for data in self.next_data]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is None:
            return None
        input = self.next_data[0]
        target = self.next_data[1]
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return [input, target]
