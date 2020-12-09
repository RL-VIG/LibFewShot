#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Wenbin Li (liwenbin.nju@gmail.com)
Date: April 9, 2019
Version: V0

Citation: 
@inproceedings{li2019DN4,
  title={Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning},
  author={Li, Wenbin and Wang, Lei and Xu, Jinglin and Huo, Jing and Gao Yang and Luo, Jiebo},
  booktitle={CVPR},
  year={2019}
}
"""


from __future__ import print_function
import argparse
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
import time
from torch import autograd
from PIL import ImageFile
import pdb
import copy
import sys
sys.dont_write_bytecode = True


# ============================ Data & Networks =====================================
import dataset.general_dataloader as GeneralDataloader
import models.network as ClassifierNet
import utils
# ==================================================================================


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='6'


model_trained = './results/Metric_Based_FSL/ADM_Conv64F_Li_Epoch_40_miniImageNet_84_84_5Way_1Shot/'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/home/liwenbin/Datasets/miniImageNet--ravi', help='/miniImageNet--ravi')
parser.add_argument('--data_name', default='miniImageNet', help='miniImageNet|StanfordDog|StanfordCar|CubBird_2011')
parser.add_argument('--mode', default='test', help='train|val|test')
parser.add_argument('--outf', default='./results/')
parser.add_argument('--resume', default=model_trained, type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--encoder_model', default='Conv64F_Li', help='Conv64F|Conv64F_Li')
parser.add_argument('--classifier_model', default='ADM', help='ProtoNet|CovaMNet|DN4|ADM')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--train_classes', type=int, default=64, help='the number of training classes')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--train_aug', action='store_true', default=True, help='Perform data augmentation or not during training')
parser.add_argument('--test_aug', action='store_true', default=False, help='Perform data augmentation or not during test')
#  Few-shot parameters  #
parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--testepisodeSize', type=int, default=1, help='one episode is taken as a mini-batch')
parser.add_argument('--epochs', type=int, default=40, help='the total number of training epoch')
parser.add_argument('--episode_train_num', type=int, default=10000, help='the total number of training episodes')
parser.add_argument('--episode_val_num', type=int, default=1000, help='the total number of evaluation episodes')
parser.add_argument('--episode_test_num', type=int, default=1000, help='the total number of testing episodes')
parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=1, help='the number of shot')
parser.add_argument('--query_num', type=int, default=15, help='the number of queries')
parser.add_argument('--aug_shot_num', type=int, default=20, help='the number of augmented support images of each class during test')
parser.add_argument('--neighbor_k', type=int, default=3, help='the number of k-nearest neighbors')
#  Other parameters   #
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--cosine', type=bool, default=True, help='using cosine annealing')
parser.add_argument('--lr_decay_epochs', type=list, default=[60,80], help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='the number of gpus')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=100, type=int, help='print frequency (default: 100)')
opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True




# ======================================= Define functions =============================================
def test(test_loader, model, criterion, epoch_index, best_prec1, F_txt):
	batch_time = utils.AverageMeter()
	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()
  

	# switch to evaluate mode
	model.eval()
	accuracies = []

	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):


		# Convert query and support images
		input_var1 = torch.cat(query_images, 0).cuda()
		input_var2 = torch.cat(support_images, 0).squeeze(0).cuda()
		input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3), input_var2.size(4))


		# Deal with the targets
		target = torch.cat(query_targets, 0).cuda()

	
		# Calculate the output
		output = model(input_var1, input_var2)
		loss = criterion(output, target)

	  
		# Measure accuracy and record loss
		prec1, _ = utils.accuracy(output, target, topk=(1,3))
		losses.update(loss.item(), target.size(0))
		top1.update(prec1[0], target.size(0))
		accuracies.append(prec1)


		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		#============== print the intermediate results ==============#
		if episode_index % opt.print_freq == 0 and episode_index != 0:

			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(test_loader), batch_time=batch_time, loss=losses, top1=top1))

			print('Test-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					epoch_index, episode_index, len(test_loader), batch_time=batch_time, loss=losses, top1=top1), file=F_txt)

		
	print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))
	print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1), file=F_txt)

	return top1.avg, losses.avg, accuracies





if __name__=='__main__':

	model = ClassifierNet.define_model(encoder_model=opt.encoder_model, classifier_model=opt.classifier_model, norm='batch',
			class_num=opt.train_classes, way_num=opt.way_num, shot_num=opt.shot_num, init_type='normal', use_gpu=opt.cuda)

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()

	# ============================================ Test phase ============================================
	# Set the save path
	opt.outf, F_txt_test = utils.set_save_test_path2(opt)
	print('========================================== Start Test ==========================================\n')
	print('========================================== Start Test ==========================================\n', file=F_txt_test)

	# Load the trained best model
	best_model_path = os.path.join(opt.resume, 'model_best.pth.tar')
	checkpoint = utils.get_resume_file(best_model_path, F_txt_test)
	epoch_index = checkpoint['epoch_index']
	best_prec1 = checkpoint['best_prec1']
	model.load_state_dict(checkpoint['model'])


	# print the parameters and architecture of the model
	print(opt)
	print(opt, file=F_txt_test)
	

	# Repeat five times
	repeat_num = 5       
	total_accuracy = 0.0
	total_h = np.zeros(repeat_num)
	for r in range(repeat_num):
		
		print('==================== The %d-th round ====================' %r)
		print('==================== The %d-th round ====================' %r, file=F_txt_test)

		# ======================================= Loaders of Datasets =======================================
		test_loader = GeneralDataloader.get_Fewshot_dataloader(opt, ['test'])


		# evaluate on validation/test set
		with torch.no_grad():
			prec1, test_loss, accuracies = test(test_loader[0], model, criterion, epoch_index, best_prec1, F_txt_test)
		test_accuracy, h = utils.mean_confidence_interval(accuracies)
		total_accuracy += test_accuracy
		total_h[r] = h

		print('Test accuracy: %f h: %f \n' %(test_accuracy, h))
		print('Test accuracy: %f h: %f \n' %(test_accuracy, h), file=F_txt_test)
	print('Mean_accuracy: %f h: %f' %(total_accuracy/repeat_num, total_h.mean()))
	print('Mean_accuracy: %f h: %f' %(total_accuracy/repeat_num, total_h.mean()), file=F_txt_test)
	print('===================================== Test is END =====================================\n')
	print('===================================== Test is END =====================================\n', file=F_txt_test)
	F_txt_test.close()