import os
from logging import getLogger
from time import time

import torch
import yaml
from torch import nn

import core.model as arch
from core.data import get_dataloader
from core.utils import init_logger, logger, prepare_device, init_seed, AverageMeter, \
    count_parameters, save_model, create_dirs, get_local_time, ModelType, TensorboardWriter, SaveType


def get_instance(module, name, config, *args):
    kwargs = dict()
    if config[name]['kwargs'] is not None:
        kwargs.update(config[name]['kwargs'])
    return getattr(module, config[name]['name'])(*args, **kwargs)


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device, self.list_ids = self._init_device(config)
        self.result_path, self.log_path, self.checkpoints_path, self.viz_path \
            = self._init_files(config)
        self.writer = TensorboardWriter(self.viz_path)
        self.train_meter, self.val_meter, self.test_meter = self._init_meter()
        self.logger = getLogger(__name__)
        self.logger.info(config)
        self.model, self.model_type = self._init_model(config)
        self.train_loader, self.val_loader, self.test_loader \
            = self._init_dataloader(config)
        self.optimizer, self.scheduler, self.from_epoch = self._init_optim(
            config)

    def train_loop(self):
        best_val_prec1 = float('-inf')
        best_test_prec1 = float('-inf')
        for epoch_idx in range(self.from_epoch + 1, self.config['epoch']):
            self.logger.info(
                '============ Train on the train set ============')
            train_prec1 = self._train(epoch_idx)
            self.logger.info(' * Prec@1 {:.3f} '.format(train_prec1))
            self.logger.info(
                '============ Validation on the val set ============')
            val_prec1 = self._validate(epoch_idx, is_test=False)
            self.logger.info(' * Prec@1 {:.3f} Best Prec1 {:.3f}'
                             .format(val_prec1, best_val_prec1))
            self.logger.info(
                '============ Testing on the test set ============')
            test_prec1 = self._validate(epoch_idx, is_test=True)
            self.logger.info(' * Prec@1 {:.3f} Best Prec1 {:.3f}'
                             .format(test_prec1, best_test_prec1))

            if val_prec1 > best_val_prec1:
                best_val_prec1 = val_prec1
                best_test_prec1 = test_prec1
                self._save_model(epoch_idx, SaveType.BEST)

            if epoch_idx != 0 and epoch_idx % self.config['save_interval'] == 0:
                self._save_model(epoch_idx, SaveType.NORMAL)

            self._save_model(epoch_idx, SaveType.LAST)

            self.scheduler.step()

    def _train(self, epoch_idx):
        self.model.train()

        meter = self.train_meter
        meter.reset()
        episode_size = 1 if self.model_type == ModelType.PRETRAIN \
            else self.config['episode_size']

        end = time()
        for batch_idx, batch in enumerate(self.train_loader):
            self.writer.set_step(epoch_idx * len(self.train_loader)
                                 + batch_idx * episode_size)

            meter.update('data_time', time() - end)

            # calculate the output
            output, prec1, loss = self.model.set_forward_loss(batch)

            # compute gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            meter.update('loss', loss.item())
            meter.update('prec1', prec1)

            # measure elapsed time
            meter.update('batch_time', time() - end)
            end = time()

            # print the intermediate results
            if batch_idx != 0 and batch_idx % self.config['log_interval'] == 0:
                info_str = ('Epoch-({}): [{}/{}]\t'
                            'Time {:.3f} ({:.3f})\t'
                            'Data {:.3f} ({:.3f})\t'
                            'Loss {:.3f} ({:.3f})\t'
                            'Prec@1 {:.3f} ({:.3f})'
                            .format(epoch_idx, batch_idx * episode_size,
                                    len(self.train_loader),
                                    meter.last('batch_time'), meter.avg(
                                        'batch_time'),
                                    meter.last('data_time'), meter.avg(
                                        'data_time'),
                                    meter.last('loss'), meter.avg('loss'),
                                    meter.last('prec1'), meter.avg('prec1'), ))
                self.logger.info(info_str)

        return meter.avg('prec1')

    def _validate(self, epoch_idx, is_test=False):
        # switch to evaluate mode
        self.model.eval()

        meter = self.test_meter if is_test else self.val_meter
        meter.reset()
        episode_size = self.config['episode_size']

        end = time()
        if self.model_type == ModelType.METRIC:
            enable_grad = False
        else:
            enable_grad = True
        with torch.set_grad_enabled(enable_grad):
            for batch_idx, batch in enumerate(
                    self.test_loader if is_test else self.val_loader):
                self.writer.set_step(epoch_idx * len(self.test_loader)
                                     + batch_idx * episode_size)

                meter.update('data_time', time() - end)

                # calculate the output
                output, prec1 = self.model.set_forward(batch)

                # measure accuracy and record loss
                meter.update('prec1', prec1)

                # measure elapsed time
                meter.update('batch_time', time() - end)
                end = time()

                if batch_idx != 0 and \
                        batch_idx % self.config['log_interval'] == 0:
                    info_str = ('Epoch-({}): [{}/{}]\t'
                                'Time {:.3f} ({:.3f})\t'
                                'Data {:.3f} ({:.3f})\t'
                                'Prec@1 {:.3f} ({:.3f})'
                                .format(epoch_idx, batch_idx * episode_size,
                                        len(self.val_loader),
                                        meter.last('batch_time'),
                                        meter.avg('batch_time'),
                                        meter.last('data_time'),
                                        meter.avg('data_time'),
                                        meter.last('prec1'),
                                        meter.avg('prec1'), ))
                    self.logger.info(info_str)

        return meter.avg('prec1')

    def _init_files(self, config):
        data_name = config['classifier']['name'].split('/')[-1]
        symlink_dir = '{}-{}-{}-{}-{}' \
            .format(config['classifier']['name'], data_name,
                    config['backbone']['name'],
                    config['way_num'], config['shot_num'])
        result_dir = symlink_dir + "-{}".format(get_local_time())
        symlink_path = os.path.join(config['result_root'], symlink_dir)
        result_path = os.path.join(config['result_root'], result_dir)

        checkpoints_path = os.path.join(result_path, 'checkpoints')
        log_path = os.path.join(result_path, 'log_files')
        viz_path = os.path.join(log_path, 'tfboard_files')
        create_dirs([result_path, log_path, checkpoints_path, viz_path])

        os.symlink(result_path, symlink_path)

        with open(os.path.join(result_path, 'config.yaml'), 'w',
                  encoding='utf-8') as fout:
            fout.write(yaml.dump(config))

        init_logger(config['log_level'], log_path,
                    config['classifier']['name'], config['backbone']['name'], )

        return result_path, log_path, checkpoints_path, viz_path

    def _init_dataloader(self, config):
        train_loader = get_dataloader(config, 'train', self.model_type)
        val_loader = get_dataloader(config, 'val', self.model_type)
        test_loader = get_dataloader(config, 'test', self.model_type)

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

        if self.config['pretrain_path'] is not None:
            self.logger.info('load pretrain model_func from {}'
                             .format(self.config['pretrain_path']))
            state_dict = torch.load(
                self.config['pretrain_path'], map_location='cpu')
            model.model_func.load_state_dict(state_dict)

        if self.config['resume']:
            resume_path = os.path.join(self.checkpoints_path, 'model_last.pth')
            self.logger.info('load the resume model checkpoints dict from {}.'
                             .format(resume_path))
            state_dict = torch.load(resume_path, map_location='cpu')['model']
            model.load_state_dict(state_dict)

        model = model.to(self.device)
        if len(self.list_ids) > 1:
            parallel_list = self.config['parallel_part']
            if parallel_list is not None:
                for parallel_part in parallel_list:
                    if hasattr(model, parallel_part):
                        setattr(model, parallel_part,
                                nn.DataParallel(getattr(model, parallel_part),
                                                device_ids=self.list_ids))

        return model, model.model_type

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
                    param_dict = {'params': sub_model.parameters()}
                    # 兼容只传了一个lr的config，在之后可以考虑规定config写法以统一
                    if isinstance(value, float):
                        param_dict.update({'lr': value})
                    # 传了一系列参数, 用缩进传入字典
                    elif isinstance(value, dict):
                        param_dict.update(value)
                    else:
                        raise Exception('Wrong config in optimizer.other')
                    params_dict_list.append(param_dict)

        params_dict_list.append({
            'params': filter(lambda p: id(p) not in params_idx, self.model.parameters())
        })
        optimizer = get_instance(
            torch.optim, 'optimizer', config, params_dict_list)
        scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config,
                                 optimizer)
        self.logger.info(optimizer)
        from_epoch = -1
        if self.config['resume']:
            resume_path = os.path.join(self.checkpoints_path, 'model_last.pth')
            self.logger.info(
                'load the optimizer, lr_scheduler and epoch checkpoints dict from {}.'
                .format(resume_path))
            all_state_dict = torch.load(resume_path, map_location='cpu')
            state_dict = all_state_dict['optimizer']
            optimizer.load_state_dict(state_dict)
            state_dict = all_state_dict['lr_scheduler']
            scheduler.load_state_dict(state_dict)
            from_epoch = all_state_dict['epoch']
            self.logger.info(
                'model resume from the epoch {}'.format(from_epoch))

        return optimizer, scheduler, from_epoch

    def _init_device(self, config):
        init_seed(config['seed'], config['deterministic'])
        device, list_ids = prepare_device(
            config['device_ids'], config['n_gpu'])
        return device, list_ids

    def _save_model(self, epoch, save_type=SaveType.NORMAL):
        save_model(self.model, self.optimizer, self.scheduler, self.checkpoints_path,
                   'model', epoch, save_type, len(self.list_ids) > 1)

        if save_type != SaveType.LAST:
            save_list = self.config['save_part']
            if save_list is not None:
                for save_part in save_list:
                    if hasattr(self.model, save_part):
                        save_model(getattr(self.model, save_part), self.optimizer,
                                   self.scheduler, self.checkpoints_path, save_part,
                                   epoch, save_type, len(self.list_ids) > 1)
                    else:
                        self.logger.warning('')

    def _init_meter(self):
        train_meter = AverageMeter('train', ['batch_time', 'data_time', 'loss', 'prec1'],
                                   self.writer)
        val_meter = AverageMeter(
            'val', ['batch_time', 'data_time', 'prec1'], self.writer)
        test_meter = AverageMeter('test', ['batch_time', 'data_time', 'prec1'],
                                  self.writer)

        return train_meter, val_meter, test_meter
