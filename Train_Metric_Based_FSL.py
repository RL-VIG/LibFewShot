#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
import sys
sys.dont_write_bytecode = True


# ============================ Data & Networks =====================================
import dataset.general_dataloader as FewShotDataloader
import models.network as FewShotNet
import utils
# ==================================================================================


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='4'



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/home/liwenbin/Datasets/miniImageNet--ravi', help='/miniImageNet')
parser.add_argument('--data_name', default='miniImageNet', help='miniImageNet|StanfordDog|StanfordCar|CubBird')
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--outf', default='./results/Metric_Based_FSL/')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--encoder_model', default='Conv64F_Li', help='Conv64F|Conv64F_Li|ResNet10|ResNet12|ResNet12_Li|ResNet18')
parser.add_argument('--classifier_model', default='ADM', help='ProtoNet | RelationNet | CovaMNet | DN4')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--train_aug', action='store_true', default=True, help='Perform data augmentation or not during training')
parser.add_argument('--test_aug', action='store_true', default=False, help='Perform data augmentation or not during test')
#  Few-shot parameters  #
parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--testepisodeSize', type=int, default=1, help='one episode is taken as a mini-batch')
parser.add_argument('--epochs', type=int, default=40, help='the total number of training epoch')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--episode_train_num', type=int, default=10000, help='the total number of training episodes')
parser.add_argument('--episode_val_num', type=int, default=1000, help='the total number of evaluation episodes')
parser.add_argument('--episode_test_num', type=int, default=1000, help='the total number of testing episodes')
parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=1, help='the number of shot')
parser.add_argument('--query_num', type=int, default=10, help='the number of queries')
parser.add_argument('--aug_shot_num', type=int, default=20, help='the number of augmented support images of each class during test')
parser.add_argument('--neighbor_k', type=int, default=3, help='the number of k-nearest neighbors')
#  Other parameters   #
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--cosine', type=bool, default=True, help='using cosine annealing')
parser.add_argument('--lr_decay_epochs', type=list, default=[60,80], help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--adam', action='store_true', default=False, help='use adam optimizer')
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
def train(train_loader, model, criterion, optimizer, epoch_index, F_txt):
	batch_time = utils.AverageMeter()
	data_time = utils.AverageMeter()
	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()


	model.train()
	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_loader):

		# Measure data loading time
		data_time.update(time.time() - end)

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


		# Compute gradients and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		# Measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


		#============== print the intermediate results ==============#
		if episode_index % opt.print_freq == 0 and episode_index != 0:

			print('Epoch-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
				'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
				'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
				'Prec@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
					epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

			print('Epoch-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
				'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
				'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
				'Prec@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
					epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1), file=F_txt)

	return top1.avg, losses.avg



def validate(val_loader, model, criterion, epoch_index, best_prec1, F_txt):
	batch_time = utils.AverageMeter()
	losses = utils.AverageMeter()
	top1 = utils.AverageMeter()
  

	# switch to evaluate mode
	model.eval()
	accuracies = []


	end = time.time()
	for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):


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

			print('Val-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
				'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
				'Prec@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

			print('Val-({0}): [{1}/{2}]\t'
				'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
				'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
				'Prec@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
					epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1), file=F_txt)

		
	print(' * Prec@1 {top1.avg:.2f} Best_prec1 {best_prec1:.2f}'.format(top1=top1, best_prec1=best_prec1))
	print(' * Prec@1 {top1.avg:.2f} Best_prec1 {best_prec1:.2f}'.format(top1=top1, best_prec1=best_prec1), file=F_txt)

	return top1.avg, losses.avg, accuracies




if __name__=='__main__':

	# save path
	opt.outf, F_txt = utils.set_save_path(opt)

	# Check if the cuda is available
	if torch.cuda.is_available() and not opt.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# ========================================== Model config ===============================================
	global best_prec1
	best_prec1 = 0
	model = FewShotNet.define_model(encoder_model=opt.encoder_model, classifier_model=opt.classifier_model, norm='batch',
			way_num=opt.way_num, shot_num=opt.shot_num, init_type='normal', use_gpu=opt.cuda)

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()
	if opt.adam:
		optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9), weight_decay=0.0005)
	else:
		optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, dampening=0.9, weight_decay=0.001)


	# optionally resume from a checkpoint
	if opt.resume:
		checkpoint = utils.get_resume_file(opt.resume, F_txt)
		opt.start_epoch = checkpoint['epoch']
		best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

	if opt.ngpu > 1:
		model = nn.DataParallel(model, range(opt.ngpu))

	# print the parameters and architecture of the model
	print(opt)
	print(opt, file=F_txt)
	print(model) 
	print(model, file=F_txt) 


	# set cosine annealing scheduler
	if opt.cosine:
		eta_min = opt.lr * (opt.lr_decay_rate ** 3)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)


	# ============================================ Training phase ========================================
	print('===================================== Training on the train set =====================================')
	print('===================================== Training on the train set =====================================', file=F_txt)

	Train_losses = []
	Val_losses = []
	Test_losses = []
	for epoch_item in range(opt.start_epoch, opt.epochs):

		print('==================== Epoch %d ====================' %epoch_item)
		print('==================== Epoch %d ====================' %epoch_item, file=F_txt)
		
	
		# Loaders of Datasets 
		opt.current_epoch = epoch_item
		train_loader, val_loader, test_loader = FewShotDataloader.get_Fewshot_dataloader(opt, ['train', 'val', 'test'])


		# train for one epoch
		prec1_train, train_loss = train(train_loader, model, criterion, optimizer, epoch_item, F_txt)
		Train_losses.append(train_loss)


		print('===================================== Validation on the val set =====================================')
		print('===================================== validation on the val set =====================================', file=F_txt)
		# evaluate on validation set
		with torch.no_grad():
			prec1_val, val_loss, _ = validate(val_loader, model, criterion, epoch_item, best_prec1, F_txt)
		Val_losses.append(val_loss)


		print('===================================== Validation on the test set =====================================')
		print('===================================== validation on the test set =====================================', file=F_txt)
		# evaluate on validation set
		with torch.no_grad():
			prec1_test, test_loss, _ = validate(test_loader, model, criterion, epoch_item, best_prec1, F_txt)
		Test_losses.append(test_loss)


		# Adjust the learning rates
		if opt.cosine:
			scheduler.step()
		else:
			utils.adjust_learning_rate(opt, optimizer, epoch_item, F_txt)


		# remember best prec@1 and save checkpoint
		is_best = prec1_val > best_prec1
		best_prec1 = max(prec1_val, best_prec1)

		# save the checkpoint
		if is_best:
			utils.save_checkpoint(
				{
					'epoch_index': epoch_item,
					'encoder_model': opt.encoder_model,
					'classifier_model': opt.classifier_model,
					'model': model.state_dict(),
					'best_prec1': best_prec1,
					'optimizer' : optimizer.state_dict(),
				}, os.path.join(opt.outf, 'model_best.pth.tar'))

		if epoch_item % 10 == 0:
			filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' %epoch_item)
			utils.save_checkpoint(
				{
					'epoch_index': epoch_item,
					'encoder_model': opt.encoder_model,
					'classifier_model': opt.classifier_model,
					'model': model.state_dict(),
					'best_prec1': best_prec1,
					'optimizer' : optimizer.state_dict(),
				}, filename)


	# ======================================= Plot Loss Curves =======================================
	utils.plot_loss_curve(opt, Train_losses, Val_losses, Test_losses)
	print('======================================== Training is END ========================================\n')
	print('======================================== Training is END ========================================\n', file=F_txt)
	F_txt.close()



	# ============================================ Test phase ============================================
	# Set the save path
	F_txt_test = utils.set_save_test_path(opt)
	print('========================================== Start Test ==========================================\n')
	print('========================================== Start Test ==========================================\n', file=F_txt_test)

	# Load the trained best model
	best_model_path = os.path.join(opt.outf, 'model_best.pth.tar')
	checkpoint = utils.get_resume_file(best_model_path, F_txt_test)
	epoch_index = checkpoint['epoch_index']
	best_prec1 = checkpoint['best_prec1']
	model.load_state_dict(checkpoint['model'])


	# print the parameters and architecture of the model
	print(opt)
	print(opt, file=F_txt_test)
	print(model) 
	print(model, file=F_txt_test) 


	# Repeat five times
	repeat_num = 5       
	total_accuracy = 0.0
	total_h = np.zeros(repeat_num)
	for r in range(repeat_num):
		
		print('==================== The %d-th round ====================' %r)
		print('==================== The %d-th round ====================' %r, file=F_txt_test)

		# ======================================= Loaders of Datasets =======================================
		opt.current_epoch = epoch_item
		_, _, test_loader = FewShotDataloader.get_Fewshot_dataloader(opt, ['train', 'val', 'test'])

		
		# evaluate on validation/test set
		with torch.no_grad():
			prec1, val_loss, accuracies = validate(test_loader, model, criterion, epoch_index, best_prec1, F_txt_test)
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
