import os
import random
from datetime import datetime
from logging import getLogger

import numpy as np
import scipy as sp
import scipy.stats
import torch


class AverageMeter(object):
    """

    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_local_time():
    """

    :return:
    """
    cur_time = datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

    return cur_time


def count_parameters(model):
    """

    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(output, target, topk=(1,)):
    """

    :param output:
    :param target:
    :param topk:
    :return:
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_confidence_interval(data, confidence=0.95):
    """

    :param data:
    :param confidence:
    :return:
    """
    a = [1.0 * np.array(data[i].cpu()) for i in range(len(data))]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
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


def prepare_device(n_gpu_use):
    """

    :param n_gpu_use:
    :return:
    """
    logger = getLogger()

    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning('the model will be performed on CPU.')
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        logger.warning('only {} are available on this machine, '
                       'but the number of the GPU in config is {}.'
                       .format(n_gpu, n_gpu_use))
        n_gpu_use = n_gpu

    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))

    return device, list_ids


def save_model(model, save_path, name, epoch, is_best=False):
    """

    :param model:
    :param save_path:
    :param name:
    :param epoch:
    :param is_best:
    :return:
    """
    if is_best:
        save_name = os.path.join(save_path, '{}_best.pth'.format(name))
        torch.save(model.state_dict(), save_name)
    else:
        save_name = os.path.join(save_path, '{}_{:0>5d}.pth'.format(name, epoch))
        torch.save(model.state_dict(), save_name)
    return save_name


def init_seed(seed=0, deterministic=False):
    """

    :param seed:
    :param deterministic:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
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


def _init_sharing_strategy(new_strategy='file_system'):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy(new_strategy)
