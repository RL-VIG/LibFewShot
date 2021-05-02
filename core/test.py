import os
from logging import getLogger
from time import time

import numpy as np
import torch
from torch import nn

import core.model as arch
from core.data import get_dataloader
from core.utils import init_logger, prepare_device, init_seed, AverageMeter, \
    count_parameters, ModelType, TensorboardWriter, mean_confidence_interval, get_local_time


def get_instance(module, name, config, *args):
    kwargs = dict()
    if config[name]['kwargs'] is not None:
        kwargs.update(config[name]['kwargs'])
    return getattr(module, config[name]['name'])(*args, **kwargs)


class Test(object):
    def __init__(self, config, result_path=None):
        self.config = config
        self.result_path = result_path
        self.device, self.list_ids = self._init_device(config)
        self.viz_path, self.state_dict_path = self._init_files(config)
        self.writer = TensorboardWriter(self.viz_path)
        self.test_meter = self._init_meter()
        self.logger = getLogger(__name__)
        self.logger.info(config)
        self.model, self.model_type = self._init_model(config)
        self.test_loader = self._init_dataloader(config)

    def test_loop(self):
        total_accuracy = 0.0
        total_h = np.zeros(self.config['test_epoch'])
        total_accuracy_vector = []

        for epoch_idx in range(self.config['test_epoch']):
            self.logger.info(
                '============ Testing on the test set ============')
            _, accuracies = self._validate(epoch_idx)
            test_accuracy, h = mean_confidence_interval(accuracies)
            self.logger.info(
                'Test Accuracy: {:.3f}\t h: {:.3f}'.format(test_accuracy, h))
            total_accuracy += test_accuracy
            total_accuracy_vector.extend(accuracies)
            total_h[epoch_idx] = h

        aver_accuracy, _ = mean_confidence_interval(total_accuracy_vector)
        self.logger.info('Aver Accuracy: {:.3f}\t Aver h: {:.3f}'
                         .format(aver_accuracy, total_h.mean()))
        self.logger.info('............Testing is end............')

    def _validate(self, epoch_idx, ):
        # switch to evaluate mode
        self.model.eval()

        meter = self.test_meter
        meter.reset()
        episode_size = self.config['episode_size']
        accuracies = []

        end = time()
        if self.model_type == ModelType.METRIC:
            enable_grad = False
        else:
            enable_grad = True

        with torch.set_grad_enabled(enable_grad):
            for episode_idx, batch in enumerate(self.test_loader):
                self.writer.set_step(epoch_idx * len(self.test_loader)
                                     + episode_idx * episode_size)

                meter.update('data_time', time() - end)

                # calculate the output
                output, acc = self.model.set_forward(batch)
                accuracies.append(acc)
                # measure accuracy and record loss
                meter.update('acc', acc)

                # measure elapsed time
                meter.update('batch_time', time() - end)
                end = time()

                if episode_idx != 0 and \
                        episode_idx % self.config['log_interval'] == 0:
                    info_str = ('Epoch-({}): [{}/{}]\t'
                                'Time {:.3f} ({:.3f})\t'
                                'Data {:.3f} ({:.3f})\t'
                                'Prec@1 {:.3f} ({:.3f})'
                                .format(epoch_idx, episode_idx, len(self.test_loader),
                                        meter.last('batch_time'),
                                        meter.avg('batch_time'),
                                        meter.last('data_time'),
                                        meter.avg('data_time'),
                                        meter.last('acc'),
                                        meter.avg('acc'), ))
                    self.logger.info(info_str)

        return meter.avg('acc'), accuracies

    def _init_files(self, config):
        if self.result_path is not None:
            result_path = self.result_path
        else:
            result_dir = '{}-{}-{}-{}-{}' \
                .format(config['classifier']['name'],
                        # you should ensure that data_root name contains its true name
                        config['data_root'].split('/')[-1],
                        config['backbone']['name'],
                        config['train_way'], config['train_shot'])
            result_path = os.path.join(config['result_root'], result_dir)

        log_path = os.path.join(result_path, 'log_files')
        viz_path = os.path.join(log_path, 'tfboard_files')

        init_logger(config['log_level'], log_path,
                    config['classifier']['name'], config['backbone']['name'],
                    is_train=False)

        state_dict_path = os.path.join(
            result_path, 'checkpoints', 'model_best.pth')

        return viz_path, state_dict_path

    def _init_dataloader(self, config):
        test_loader = get_dataloader(config, 'test', self.model_type)

        return test_loader

    def _init_model(self, config):
        emb_func = get_instance(arch, 'backbone', config)
        model = get_instance(arch, 'classifier', config,
                             config['train_way'],
                             config['train_shot'] * config['augment_times'],
                             config['train_query'],
                             emb_func, self.device)

        self.logger.info(model)
        self.logger.info(count_parameters(model))

        self.logger.info(
            'load the state dict from {}.'.format(self.state_dict_path))
        state_dict = torch.load(self.state_dict_path, map_location='cpu')
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

    def _init_device(self, config):
        init_seed(config['seed'], config['deterministic'])
        device, list_ids = prepare_device(
            config['device_ids'], config['n_gpu'])
        return device, list_ids

    def _init_meter(self):
        test_meter = AverageMeter('test', ['batch_time', 'data_time', 'acc'],
                                  self.writer)

        return test_meter
