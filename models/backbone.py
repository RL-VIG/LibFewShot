import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import math
import pdb
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm



##############################################################################
# Embedding backbone networks: Shallow Conv64F 
##############################################################################

class Conv64F(nn.Module):
	'''
		Four convolutional blocks network, each of which consists of a Covolutional layer, 
		a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
		Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.
		
		Input:  3 * 84 *84
		Output: 64 * 5 * 5
	'''
	def __init__(self):
		super(Conv64F, self).__init__()


		self.features = nn.Sequential(                              # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*42*42


			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*21*21


			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*10*10


			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*5*5
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)
	  

	def forward(self, x):
		
		out = self.features(x)  # 64 * 5 * 5                                           
		
		return out



class Conv64F_Li(nn.Module):
	'''
		Four convolutional blocks network, each of which consists of a Covolutional layer, 
		a Batch Normalizaiton layer, a LeakyReLU layer and a Maxpooling layer.
		Used in the original DN4 and CovaMNet: https://github.com/WenbinLee/DN4.git.
		
		Input:  3 * 84 *84
		Output: 64 * 21 * 21
	'''
	def __init__(self):
		super(Conv64F_Li, self).__init__()


		self.features = nn.Sequential(                              # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*42*42


			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),                  # 64*21*21


			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, True),                               # 64*21*21


			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, True)                                # 64*21*21
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)
	  

	def forward(self, x):
		
		out = self.features(x)  # 64 * 21 * 21                                           
		
		return out



##############################################################################
# Embedding backbone networks: Other ResNet Variants 
# Referred to https://github.com/wyharveychen/CloserLookFewShot.git

# Input:  3 * 224 * 224
# Output: 512 * 7 * 7     ('ResNet10', 'ResNet18', 'ResNet34')
# Output: 512 * 14 * 14   ('ResNet10_Li', 'ResNet18_Li', 'ResNet34_Li')
# Output: 2048 * 7 * 7  ('ResNet50', 'ResNet101')
##############################################################################

def init_layer(L):
	# Initialization using fan-in
	if isinstance(L, nn.Conv2d):
		n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
		L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
	elif isinstance(L, nn.BatchNorm2d):
		L.weight.data.fill_(1)
		L.bias.data.fill_(0)


class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()
		
	def forward(self, x):        
		return x.view(x.size(0), -1)


# Simple ResNet Block
class SimpleBlock(nn.Module):
	maml = False #Default
	def __init__(self, indim, outdim, half_res):
		super(SimpleBlock, self).__init__()
		self.indim = indim
		self.outdim = outdim
		if self.maml:
			self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
			self.BN1 = BatchNorm2d_fw(outdim)
			self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
			self.BN2 = BatchNorm2d_fw(outdim)
		else:
			self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
			self.BN1 = nn.BatchNorm2d(outdim)
			self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
			self.BN2 = nn.BatchNorm2d(outdim)
		self.relu1 = nn.ReLU(inplace=True)
		self.relu2 = nn.ReLU(inplace=True)

		self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

		self.half_res = half_res

		# if the input number of channels is not equal to the output, then need a 1x1 convolution
		if indim!=outdim:
			if self.maml:
				self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
				self.BNshortcut = BatchNorm2d_fw(outdim)
			else:
				self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
				self.BNshortcut = nn.BatchNorm2d(outdim)

			self.parametrized_layers.append(self.shortcut)
			self.parametrized_layers.append(self.BNshortcut)
			self.shortcut_type = '1x1'
		else:
			self.shortcut_type = 'identity'

		for layer in self.parametrized_layers:
			init_layer(layer)

	def forward(self, x):
		out = self.C1(x)
		out = self.BN1(out)
		out = self.relu1(out)
		out = self.C2(out)
		out = self.BN2(out)
		short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
		out = out + short_out
		out = self.relu2(out)
		return out



# Bottleneck block
class BottleneckBlock(nn.Module):
	maml = False #Default
	def __init__(self, indim, outdim, half_res):
		super(BottleneckBlock, self).__init__()
		bottleneckdim = int(outdim/4)
		self.indim = indim
		self.outdim = outdim
		if self.maml:
			self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
			self.BN1 = BatchNorm2d_fw(bottleneckdim)
			self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
			self.BN2 = BatchNorm2d_fw(bottleneckdim)
			self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
			self.BN3 = BatchNorm2d_fw(outdim)
		else:
			self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
			self.BN1 = nn.BatchNorm2d(bottleneckdim)
			self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
			self.BN2 = nn.BatchNorm2d(bottleneckdim)
			self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
			self.BN3 = nn.BatchNorm2d(outdim)

		self.relu = nn.ReLU()
		self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
		self.half_res = half_res


		# if the input number of channels is not equal to the output, then need a 1x1 convolution
		if indim!=outdim:
			if self.maml:
				self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
			else:
				self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

			self.parametrized_layers.append(self.shortcut)
			self.shortcut_type = '1x1'
		else:
			self.shortcut_type = 'identity'

		for layer in self.parametrized_layers:
			init_layer(layer)


	def forward(self, x):

		short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
		out = self.C1(x)
		out = self.BN1(out)
		out = self.relu(out)
		out = self.C2(out)
		out = self.BN2(out)
		out = self.relu(out)
		out = self.C3(out)
		out = self.BN3(out)
		out = out + short_out

		out = self.relu(out)
		return out



class ResNet_224(nn.Module):
	maml = False #Default
	def __init__(self,block,list_of_num_layers, list_of_out_dims, No_pool=False, flatten = True):
		# list_of_num_layers specifies number of layers in each stage
		# list_of_out_dims specifies number of output channel for each stage
		super(ResNet_224,self).__init__()
		assert len(list_of_num_layers)==4, 'Can have only four stages'
		if self.maml:
			conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
											   bias=False)
			bn1 = BatchNorm2d_fw(64)
		else:
			conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
											   bias=False)
			bn1 = nn.BatchNorm2d(64)

		relu = nn.ReLU()
		pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		init_layer(conv1)
		init_layer(bn1)

		if not No_pool:
			trunk = [conv1, bn1, relu, pool1]
		else:
			trunk = [conv1, bn1, relu]


		indim = 64
		for i in range(4):

			for j in range(list_of_num_layers[i]):
				half_res = (i>=1) and (j==0)
				B = block(indim, list_of_out_dims[i], half_res)
				trunk.append(B)
				indim = list_of_out_dims[i]

		if flatten:
			avgpool = nn.AvgPool2d(7)
			trunk.append(avgpool)
			trunk.append(Flatten())
			self.final_feat_dim = indim
		else:
			self.final_feat_dim = [ indim, 7, 7]

		self.trunk = nn.Sequential(*trunk)

	def forward(self,x):
		out = self.trunk(x)
		return out



##############################################################################
# Embedding backbone networks: ResNet12 
# Referred to https://github.com/WangYueFt/rfs.git

# Input:  3 * 84 * 84
# Output: 640 * 5 * 5     ('ResNet12', 'SeResNet12')
# Output: 640 * 21 * 21   ('ResNet12_Li', 'SeResNet12_Li')
##############################################################################

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
				nn.Linear(channel, channel // reduction),
				nn.ReLU(inplace=True),
				nn.Linear(channel // reduction, channel),
				nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y


class DropBlock(nn.Module):
	def __init__(self, block_size):
		super(DropBlock, self).__init__()

		self.block_size = block_size
		#self.gamma = gamma
		#self.bernouli = Bernoulli(gamma)

	def forward(self, x, gamma):
		# shape: (bsize, channels, height, width)

		if self.training:
			batch_size, channels, height, width = x.shape
			
			bernoulli = Bernoulli(gamma)
			mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
			block_mask = self._compute_block_mask(mask)
			countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
			count_ones = block_mask.sum()

			return block_mask * x * (countM / count_ones)
		else:
			return x

	def _compute_block_mask(self, mask):
		left_padding = int((self.block_size-1) / 2)
		right_padding = int(self.block_size / 2)
		
		batch_size, channels, height, width = mask.shape
		#print ("mask", mask[0][0])
		non_zero_idxs = mask.nonzero()
		nr_blocks = non_zero_idxs.shape[0]

		offsets = torch.stack(
			[
				torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
				torch.arange(self.block_size).repeat(self.block_size), #- left_padding
			]
		).t().cuda()
		offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
		
		if nr_blocks > 0:
			non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
			offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
			offsets = offsets.long()

			block_idxs = non_zero_idxs + offsets
			#block_idxs += left_padding
			padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
			padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
		else:
			padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
			
		block_mask = 1 - padded_mask#[:height, :width]
		return block_mask
	

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
				 block_size=1, No_pool=False, use_se=False):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.LeakyReLU(0.1)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv3x3(planes, planes)
		self.bn3 = nn.BatchNorm2d(planes)
		self.maxpool = nn.MaxPool2d(stride)
		self.downsample = downsample
		self.stride = stride
		self.drop_rate = drop_rate
		self.num_batches_tracked = 0
		self.drop_block = drop_block
		self.block_size = block_size
		self.DropBlock = DropBlock(block_size=self.block_size)
		self.use_se = use_se
		self.No_pool = No_pool
		if self.use_se:
			self.se = SELayer(planes, 4)

	def forward(self, x):
		self.num_batches_tracked += 1

		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)
		if self.use_se:
			out = self.se(out)

		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		if not self.No_pool:
			out = self.maxpool(out)

		if self.drop_rate > 0:
			if self.drop_block == True:
				feat_size = out.size()[2]
				keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
				gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
				out = self.DropBlock(out, gamma=gamma)
			# else:
			# 	out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

		return out


class ResNet_84(nn.Module):

	def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, flatten=False, drop_rate=0.0,
				 dropblock_size=5, num_classes=-1, No_pool=False, use_se=False):
		super(ResNet_84, self).__init__()

		self.inplanes = 3
		self.use_se = use_se
		self.layer1 = self._make_layer(block, n_blocks[0], 64,
									   stride=2, drop_rate=drop_rate)
		self.layer2 = self._make_layer(block, n_blocks[1], 160,
									   stride=2, drop_rate=drop_rate)
		self.layer3 = self._make_layer(block, n_blocks[2], 320,
									   stride=2, drop_rate=drop_rate, No_pool=No_pool, drop_block=True, block_size=dropblock_size)
		self.layer4 = self._make_layer(block, n_blocks[3], 640,
									   stride=2, drop_rate=drop_rate, No_pool=No_pool, drop_block=True, block_size=dropblock_size)
		if avg_pool:
			# self.avgpool = nn.AvgPool2d(5, stride=1)
			self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.keep_prob = keep_prob
		self.keep_avg_pool = avg_pool
		self.flatten = flatten
		# self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
		self.drop_rate = drop_rate
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.num_classes = num_classes
		if self.num_classes > 0:
			self.classifier = nn.Linear(640, self.num_classes)
			self.rot_classifier = nn.Linear(self.num_classes, 4)

	def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, No_pool=False, drop_block=False, block_size=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		if n_block == 1:
			layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, No_pool, self.use_se)
		else:
			layer = block(self.inplanes, planes, stride, downsample, drop_rate, No_pool, self.use_se)
		layers.append(layer)
		self.inplanes = planes * block.expansion

		for i in range(1, n_block):
			if i == n_block - 1:
				layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
							  block_size=block_size,  No_pool=No_pool, use_se=self.use_se)
			else:
				layer = block(self.inplanes, planes, drop_rate=drop_rate, No_pool=No_pool, use_se=self.use_se)
			layers.append(layer)

		return nn.Sequential(*layers)

	def forward(self, x, is_feat=False, rot=False):
		x = self.layer1(x)
		f0 = x
		x = self.layer2(x)
		f1 = x
		x = self.layer3(x)
		f2 = x
		x = self.layer4(x)
		f3 = x
		if self.keep_avg_pool:
			x = self.avgpool(x)

		if self.flatten:
			x = x.view(x.size(0), -1)
		feat = x

		if self.num_classes > 0:
			x = self.classifier(x)

		if rot:
			x_rot = self.rot_classifier(x)
			return [f0, f1, f2, f3, feat], (x, x_rot)

		if is_feat:
			return [f0, f1, f2], x
		else:
			return x


def ResNet12(keep_prob=1.0, avg_pool=False, flatten=False, **kwargs):
	"""Constructs a ResNet-12 model"""
	return ResNet_84(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, 
					flatten=flatten, **kwargs)

def SeResNet12(keep_prob=1.0, avg_pool=False, flatten=False, **kwargs):
	"""Constructs a SEResNet-12 model."""
	return ResNet_84(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, 
					flatten=flatten, use_se=True, **kwargs)

def ResNet10(flatten = False):
	return ResNet_224(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18(flatten = False):
	return ResNet_224(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34(flatten = False):
	return ResNet_224(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50(flatten = False):
	return ResNet_224(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101(flatten = False):
	return ResNet_224(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)



##############################################################################
# If you want to obtain more local descriptors, you can set 'No_pool=True'.
##############################################################################

def ResNet12_Li(keep_prob=1.0, avg_pool=False, No_pool=True, flatten=False, **kwargs):
	"""Constructs a ResNet-12 model"""
	return ResNet_84(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, 
					No_pool=No_pool, flatten=flatten, **kwargs)

def SeResNet12_Li(keep_prob=1.0, avg_pool=False, No_pool=True, flatten=False, **kwargs):
	"""Constructs a SEResNet-12 model."""
	return ResNet_84(BasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool,  
					No_pool=No_pool, flatten=flatten, use_se=True, **kwargs)

def ResNet10_Li(No_pool=True, flatten = False):
	return ResNet_224(SimpleBlock, [1,1,1,1],[64,128,256,512], No_pool, flatten)

def ResNet18_Li(No_pool=True, flatten = False):
	return ResNet_224(SimpleBlock, [2,2,2,2],[64,128,256,512], No_pool, flatten)

def ResNet34_Li(No_pool=True, flatten = False):
	return ResNet_224(SimpleBlock, [3,4,6,3],[64,128,256,512], No_pool, flatten)






