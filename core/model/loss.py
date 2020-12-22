from __future__ import print_function

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from PIL import ImageFile
from torch import optim

from datasets import ImageFolder
from models import ALTNet
from utils import AverageMeter, print_func, save_model
from utils import accuracy
from utils import prepare_device

sys.dont_write_bytecode = True

mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]


def train(train_loader, model, criterion, optimizer, epoch_index, device, fout_file, image2level, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        way_num = len(support_images)
        shot_num = len(support_images[0])
        query_input = torch.cat(query_images, 0)
        query_targets = torch.cat(query_targets, 0)
        # support_targets = torch.cat(support_targets, 0)

        if image2level == 'image2task':
            image_list = []
            for images in support_images:
                image_list.extend(images)
            support_input = [torch.cat(image_list, 0)]
        else:
            raise RuntimeError

        query_input = query_input.to(device)
        query_targets = query_targets.to(device)
        support_input = [item.to(device) for item in support_input]
        # support_targets = support_targets.to(device)

        # calculate the output
        _, output, _ = model(query_input, support_input)
        output = torch.sum(output.view(-1, way_num, shot_num), dim=2)
        loss = criterion(output, query_targets)

        # compute gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))
        losses.update(loss.item(), query_input.size(0))
        top1.update(prec1[0], query_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print the intermediate results
        if episode_index % print_freq == 0 and episode_index != 0:
            info_str = ('Eposide-({0}): [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                        .format(epoch_index, episode_index, len(train_loader), batch_time=batch_time,
                                data_time=data_time, loss=losses, top1=top1))
            print_func(info_str, fout_file)


def validate(val_loader, model, criterion, epoch_index, device, fout_file, image2level, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    accuracies = []

    end = time.time()
    with torch.no_grad():
        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):

            way_num = len(support_images)
            shot_num = len(support_images[0])
            query_input = torch.cat(query_images, 0)
            query_targets = torch.cat(query_targets, 0)

            if image2level == 'image2task':
                image_list = []
                for images in support_images:
                    image_list.extend(images)
                support_input = [torch.cat(image_list, 0)]
            else:
                raise RuntimeError

            query_input = query_input.to(device)
            query_targets = query_targets.to(device)
            support_input = [item.to(device) for item in support_input]
            # support_targets = support_targets.to(device)

            # calculate the output
            _, output, _ = model(query_input, support_input)
            output = torch.mean(output.view(-1, way_num, shot_num), dim=2)
            loss = criterion(output, query_targets)

            # measure accuracy and record loss
            prec1, _ = accuracy(output, query_targets, topk=(1, 3))
            losses.update(loss.item(), query_input.size(0))
            top1.update(prec1[0], query_input.size(0))
            accuracies.append(prec1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print the intermediate results
            if episode_index % print_freq == 0 and episode_index != 0:
                info_str = ('Test-({0}): [{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                            .format(epoch_index, episode_index, len(val_loader), batch_time=batch_time,
                                    loss=losses, top1=top1))
                print_func(info_str, fout_file)
    return top1.avg


def main(config):
    result_name = '{}_{}_{}way_{}shot'.format(config['data_name'], config['arch']['base_model'],
                                              config['general']['way_num'], config['general']['shot_num'], )
    save_path = os.path.join(config['general']['save_root'], result_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fout_path = os.path.join(save_path, 'train_info.txt')
    fout_file = open(fout_path, 'a+')
    with open(os.path.join(save_path, 'config.json'), 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=True)
    print_func(config, fout_file)

    train_trsfms = transforms.Compose([
        transforms.Resize((config['general']['image_size'], config['general']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_trsfms = transforms.Compose([
        transforms.Resize((config['general']['image_size'], config['general']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    model = ALTNet(**config['arch'])
    print_func(model, fout_file)

    optimizer = optim.Adam(model.parameters(), lr=config['train']['optim_lr'])

    if config['train']['lr_scheduler']['name'] == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, **config['train']['lr_scheduler']['args'])
    elif config['train']['lr_scheduler']['name'] == 'MultiStepLR':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, **config['train']['lr_scheduler']['args'])
    else:
        raise RuntimeError

    if config['train']['loss']['name'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(**config['train']['loss']['args'])
    else:
        raise RuntimeError

    device, _ = prepare_device(config['n_gpu'])
    model = model.to(device)
    criterion = criterion.to(device)

    best_val_prec1 = 0
    best_test_prec1 = 0
    for epoch_index in range(config['train']['epochs']):
        print_func('{} Epoch {} {}'.format('=' * 35, epoch_index, '=' * 35), fout_file)
        train_dataset = ImageFolder(
            data_root=config['general']['data_root'], mode='train', episode_num=config['train']['episode_num'],
            way_num=config['general']['way_num'], shot_num=config['general']['shot_num'],
            query_num=config['general']['query_num'], transform=train_trsfms,
        )
        val_dataset = ImageFolder(
            data_root=config['general']['data_root'], mode='val', episode_num=config['test']['episode_num'],
            way_num=config['general']['way_num'], shot_num=config['general']['shot_num'],
            query_num=config['general']['query_num'], transform=val_trsfms,
        )
        test_dataset = ImageFolder(
            data_root=config['general']['data_root'], mode='test', episode_num=config['test']['episode_num'],
            way_num=config['general']['way_num'], shot_num=config['general']['shot_num'],
            query_num=config['general']['query_num'], transform=val_trsfms,
        )

        print_func('The num of the train_dataset: {}'.format(len(train_dataset)), fout_file)
        print_func('The num of the val_dataset: {}'.format(len(val_dataset)), fout_file)
        print_func('The num of the test_dataset: {}'.format(len(test_dataset)), fout_file)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['train']['batch_size'], shuffle=True,
            num_workers=config['general']['workers_num'], drop_last=True, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['test']['batch_size'], shuffle=True,
            num_workers=config['general']['workers_num'], drop_last=True, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['test']['batch_size'], shuffle=True,
            num_workers=config['general']['workers_num'], drop_last=True, pin_memory=True
        )

        # train for 5000 episodes in each epoch
        print_func('============ Train on the train set ============', fout_file)
        train(train_loader, model, criterion, optimizer, epoch_index, device, fout_file,
              config['general']['image2level'],
              config['general']['print_freq'])

        print_func('============ Validation on the val set ============', fout_file)
        val_prec1 = validate(val_loader, model, criterion, epoch_index, device, fout_file,
                             config['general']['image2level'],
                             config['general']['print_freq'])
        print_func(' * Prec@1 {:.3f} Best Prec1 {:.3f}'.format(val_prec1, best_val_prec1), fout_file)

        print_func('============ Testing on the test set ============', fout_file)
        test_prec1 = validate(test_loader, model, criterion, epoch_index, device, fout_file,
                              config['general']['image2level'],
                              config['general']['print_freq'])
        print_func(' * Prec@1 {:.3f} Best Prec1 {:.3f}'.format(test_prec1, best_test_prec1), fout_file)

        if val_prec1 > best_val_prec1:
            best_val_prec1 = val_prec1
            best_test_prec1 = test_prec1
            save_model(model, save_path, config['data_name'], epoch_index, is_best=True)

        if epoch_index % config['general']['save_freq'] == 0 and epoch_index != 0:
            save_model(model, save_path, config['data_name'], epoch_index, is_best=False)

        lr_scheduler.step()

    print_func('............Training is end............', fout_file)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    parser = argparse.ArgumentParser(description='ATL_Net in PyTorch')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        config = json.load(open(os.path.abspath(args.config)))
    else:
        raise AssertionError("config file need to be specified")
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config)