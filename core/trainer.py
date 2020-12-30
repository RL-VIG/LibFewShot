import os
from logging import getLogger
from time import time

import torch
import yaml

import core.model as arch
from core.data import get_dataloader
from core.utils import init_logger, prepare_device, init_seed, AverageMeter, \
    count_parameters, save_model, create_dirs
from core.utils.utils import _init_sharing_strategy


def get_instance(module, name, config, *args):
    kwargs = dict()
    if config[name]['kwargs'] is not None:
        kwargs.update(config[name]['kwargs'])
    return getattr(module, config[name]['name'])(*args, **kwargs)


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device, self.list_ids = self._init_device(config)
        self.result_path, self.log_path, self.checkpoints_path \
            = self._init_files(config)
        self.logger = getLogger(__name__)
        self.logger.info(config)
        self.train_loader, self.val_loader, self.test_loader \
            = self._init_dataloader(config)
        self.model = self._init_model(config)
        self.optimizer, self.scheduler = self._init_optim(config)

        self.model = self.model.to(self.device)

    def train_loop(self):
        best_val_prec1 = float('-inf')
        best_test_prec1 = float('-inf')
        for epoch_idx in range(self.config['epoch']):
            self.logger.info('============ Train on the train set ============')
            train_prec1 = self._train(epoch_idx)
            self.logger.info('============ Validation on the val set ============')
            val_prec1 = self._validate(epoch_idx, is_test=False)
            self.logger.info('============ Testing on the test set ============')
            test_prec1 = self._validate(epoch_idx, is_test=True)

            if val_prec1 > best_test_prec1:
                best_val_prec1 = val_prec1
                best_test_prec1 = test_prec1
                self._save_model(epoch_idx, is_best=True)

            if epoch_idx != 0 and epoch_idx % self.config['save_interval'] == 0:
                self._save_model(epoch_idx, is_best=False)

    def _train(self, epoch_idx):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time()
        for episode_idx, batch in enumerate(self.train_loader):
            data_time.update(time() - end)

            # calculate the output
            output, prec1, loss = self.model.set_forward_loss(batch)

            # compute gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss

            losses.update(loss.item())
            top1.update(prec1[0])

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            # print the intermediate results
            if episode_idx != 0 and episode_idx % self.config['log_interval'] == 0:
                info_str = ('Epoch-({0}): [{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                            .format(epoch_idx, episode_idx, len(self.train_loader),
                                    batch_time=batch_time, data_time=data_time,
                                    loss=losses, top1=top1))
                self.logger.info(info_str)

        return top1.avg

    def _validate(self, epoch_idx, is_test=False):
        # switch to evaluate mode
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()

        prec1_list = []

        end = time()
        with torch.no_grad():
            for episode_idx, batch in enumerate(
                    self.test_loader if is_test else self.val_loader):
                data_time.update(time() - end)

                # calculate the output
                output, prec1 = self.model.set_forward(batch)

                # measure accuracy and record loss
                top1.update(prec1[0])
                prec1_list.append(prec1)

                # measure elapsed time
                batch_time.update(time() - end)
                end = time()

                if episode_idx != 0 and episode_idx % self.config['log_interval'] == 0:
                    info_str = ('Epoch-({0}): [{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                                .format(epoch_idx, episode_idx, len(self.train_loader),
                                        batch_time=batch_time, data_time=data_time,
                                        top1=top1))
                    self.logger.info(info_str)

        return top1.avg

    def _init_files(self, config):
        result_dir = '{}-{}-{}-{}' \
            .format(config['classifier']['name'], config['backbone']['name'],
                    config['way_num'], config['shot_num'])
        result_path = os.path.join(config['result_root'], result_dir)

        log_path = os.path.join(result_path, 'log')
        checkpoints_path = os.path.join(result_path, 'checkpoints')
        create_dirs([result_path, log_path, checkpoints_path])

        with open(os.path.join(result_path, 'config.yaml'), 'w',
                  encoding='utf-8') as fout:
            fout.write(yaml.dump(config))

        init_logger(config['log_level'], log_path,
                    config['classifier']['name'], config['backbone']['name'], )

        return result_path, log_path, checkpoints_path

    def _init_dataloader(self, config):
        train_loader = get_dataloader(config, 'train')
        val_loader = get_dataloader(config, 'val')
        test_loader = get_dataloader(config, 'test')

        _init_sharing_strategy()

        return train_loader, val_loader, test_loader

    def _init_model(self, config):
        model_func = get_instance(arch, 'backbone', config)
        model = get_instance(arch, 'classifier', config,
                             config['way_num'],
                             config['shot_num'] * config['augment_times'],
                             config['query_num'],
                             model_func, self.device)

        self.logger.info(model)
        self.logger.info(count_parameters(model))

        return model

    def _init_optim(self, config):
        params_idx = []
        params_dict_list = []
        if config['optimizer']['other'] is not None:
            for key, value in config['optimizer']['other'].items():
                sub_model = getattr(self.model, key)
                params_idx.extend(list(map(id, sub_model.parameters())))
                if value is None:
                    for p in sub_model.parameters():
                        p.requires_grad = False
                else:
                    params_dict_list.append({
                        'params': sub_model.parameters(), 'lr': value
                    })

        params_dict_list.append({
            'params': filter(lambda p: id(p) not in params_idx, self.model.parameters())
        })
        optimizer = get_instance(torch.optim, 'optimizer', config, params_dict_list)
        scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config,
                                 optimizer)

        return optimizer, scheduler

    def _init_device(self, config):
        init_seed(config['seed'], config['deterministic'])
        device, list_ids = prepare_device(config['n_gpu'])
        return device, list_ids

    def _save_model(self, epoch, is_best=False):
        save_model(self.model, self.checkpoints_path, 'model', epoch, is_best)
        save_list = self.config['save_part']
        if save_list is not None:
            for save_part in save_list:
                if hasattr(self.model, save_part):
                    save_model(getattr(self.model, save_part), self.checkpoints_path,
                               save_part, epoch, is_best)
                else:
                    self.logger.warning('')
