import torch
import os
import pdb
import scipy as sp
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def adjust_learning_rate(opt, optimizer, epoch, F_txt):
	"""Sets the learning rate to the initial LR decayed by 2 every 10 epoches"""
	if opt.classifier_model == 'Baseline':
		lr = opt.lr * (0.5 ** (epoch // 30))
	else:
		lr = opt.lr * (0.1 ** (epoch // 10))
	print('Learning rate: %f' %lr)
	print('Learning rate: %f' %lr, file=F_txt)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate2(opt, optimizer, epoch, F_txt):
	"""Sets the learning rate to the initial LR decayed by decay rate every steep step"""
	steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
	if steps > 0:
		new_lr = opt.lr * (opt.lr_decay_rate ** steps)
		print('Learning rate: %f' %new_lr)
		print('Learning rate: %f' %new_lr, file=F_txt)
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr


def count_parameters(model):
	"""Count the total number of parameters in one model"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

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


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
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
	a = [1.0*np.array(data[i].cpu()) for i in range(len(data))]
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
	return m,h



def set_save_path(opt):
	'''
		Settings of the save path
	'''
	if opt.classifier_model in ['Baseline', 'Baseline_plus', 'SKD']:
		opt.outf = opt.outf + opt.classifier_model + '_' + opt.encoder_model + '_' + opt.New_classifier_model + '_' + 'Epoch_' + str(opt.epochs) + '_' +\
					opt.data_name + '_' + str(opt.imageSize) + '_' + str(opt.imageSize) + '_' + str(opt.way_num)+'Way_'+str(opt.shot_num)+'Shot'
	else:
		opt.outf = opt.outf + opt.classifier_model + '_' + opt.encoder_model + '_' + 'Epoch_' + str(opt.epochs) + '_' +\
					opt.data_name + '_' + str(opt.imageSize) + '_' + str(opt.imageSize) + '_' + str(opt.way_num)+'Way_'+str(opt.shot_num)+'Shot'

	if not os.path.exists(opt.outf):
		os.makedirs(opt.outf)

	if torch.cuda.is_available() and not opt.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# save the opt and results to txt file
	txt_save_path = os.path.join(opt.outf, 'opt_results.txt')
	F_txt = open(txt_save_path, 'a+')

	return opt.outf, F_txt




def set_save_test_path(opt, finetune=False):
	'''
		Settings of the save path
	'''

	if not os.path.exists(opt.outf):
		os.makedirs(opt.outf)

	if torch.cuda.is_available() and not opt.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# save the opt and results to txt file
	if finetune:
		txt_save_path = os.path.join(opt.outf, 'Test_Finetune_results.txt')
	else:
		txt_save_path = os.path.join(opt.outf, 'Test_results.txt')
	F_txt_test = open(txt_save_path, 'a+')

	return F_txt_test



def set_save_test_path2(opt, finetune=False):
	'''
		Settings of the save path
	'''
	
	if not str(opt.resume).endswith('/'):
		opt.outf = opt.resume + '/'
	else:
		opt.outf = opt.resume

	if not os.path.exists(opt.outf):
		os.makedirs(opt.outf)

	if torch.cuda.is_available() and not opt.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# save the opt and results to txt file
	if finetune:
		txt_save_path = os.path.join(opt.outf, 'Test_Finetune_results_New.txt')
	else:
		txt_save_path = os.path.join(opt.outf, 'Test_results_New.txt')
	F_txt_test = open(txt_save_path, 'a+')

	return opt.outf, F_txt_test



def get_resume_file(checkpoint_dir, F_txt):

	if os.path.isfile(checkpoint_dir):
		print("=> loading checkpoint '{}'".format(checkpoint_dir))
		print("=> loading checkpoint '{}'".format(checkpoint_dir), file=F_txt)
		checkpoint = torch.load(checkpoint_dir)
		print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_dir, checkpoint['epoch_index']))
		print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_dir, checkpoint['epoch_index']), file=F_txt)

		return checkpoint
	else:
		print("=> no checkpoint found at '{}'".format(checkpoint_dir))
		print("=> no checkpoint found at '{}'".format(checkpoint_dir), file=F_txt)

		return None


def plot_loss_curve(opt, train_loss, val_loss, test_loss=None):

	if test_loss:
		train_loss = np.array(train_loss)
		val_loss = np.array(val_loss)
		test_loss = np.array(test_loss)


		# Save lossed to txt
		np.savetxt(os.path.join(opt.outf, 'train_loss.txt'), train_loss)
		np.savetxt(os.path.join(opt.outf, 'val_loss.txt'), val_loss)
		np.savetxt(os.path.join(opt.outf, 'test_loss.txt'), test_loss)

		# Plot the loss curves
		fig, ax = plt.subplots()
		ax.plot(range(0, opt.epochs), train_loss, label='Train loss')
		ax.plot(range(0, opt.epochs), val_loss, label='Val loss')
		ax.plot(range(0, opt.epochs), test_loss, label='Test loss')
		legend = ax.legend(loc='upper right', fontsize='medium')
		plt.savefig(os.path.join(opt.outf, 'Loss.png'), bbox_inches='tight')
		# plt.show()
	else:
		train_loss = np.array(train_loss)
		val_loss = np.array(val_loss)


		# Save lossed to txt
		np.savetxt(os.path.join(opt.outf, 'train_loss.txt'), train_loss)
		np.savetxt(os.path.join(opt.outf, 'val_loss.txt'), val_loss)


		# Plot the loss curves
		fig, ax = plt.subplots()
		ax.plot(range(0, opt.epochs), train_loss, label='Train loss')
		ax.plot(range(0, opt.epochs), val_loss, label='Val loss')
		legend = ax.legend(loc='upper right', fontsize='medium')
		plt.savefig(os.path.join(opt.outf, 'Loss.png'), bbox_inches='tight')
		# plt.show()






