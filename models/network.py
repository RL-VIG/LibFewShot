import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools
import random
import pdb
import copy
import sys
sys.dont_write_bytecode = True



# ============================ Backbone & Classifier ===============================
import models.backbone as backbone
import models.classifier as classifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# ==================================================================================
''' 
	All models consist of two parts: backbone module and classifier module.
'''


###############################################################################
# Functions
###############################################################################

encoder_dict = dict(
			Conv64F       = backbone.Conv64F,
			Conv64F_Li    = backbone.Conv64F_Li,
			ResNet10      = backbone.ResNet10,
			ResNet12      = backbone.ResNet12,
			SeResNet12    = backbone.SeResNet12,
			ResNet18      = backbone.ResNet18,
			ResNet34      = backbone.ResNet34,
			ResNet10_Li   = backbone.ResNet10_Li,
			ResNet12_Li   = backbone.ResNet12_Li,
			SeResNet12_Li = backbone.SeResNet12_Li,
			ResNet18_Li   = backbone.ResNet18_Li,
			ResNet34_Li   = backbone.ResNet34_Li,
			ResNet50      = backbone.ResNet50,
			ResNet101     = backbone.ResNet101) 


classifier_dict = dict(
			Baseline      = classifier.Baseline_Metric,
			Baseline_plus = classifier.Baseline_Metric,
			RFS           = classifier.Baseline_Metric,
			SKD           = classifier.Baseline_Metric,
			ProtoNet      = classifier.Prototype_Metric,
			RelationNet   = classifier.LearnToCompare_Metric,
			CovaMNet      = classifier.Covariance_Metric,
			DN4           = classifier.ImgtoClass_Metric,
			ADM           = classifier.ADM_Metric) 



def weights_init_normal(m):
	classname = m.__class__.__name__
	# pdb.set_trace()
	# print(classname)
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
	print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	
	print('Total number of parameters: %d' % num_params)



def define_model(pretrained=False, model_root=None, encoder_model='Conv64F', classifier_model='ProtoNet', norm='batch', init_type='normal', use_gpu=True, **kwargs):
	model = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if   classifier_model in ['Baseline', 'Baseline_plus']:
		model = General_model(encoder_model=encoder_model, classifier_model=classifier_model, **kwargs)

	elif classifier_model in ['ProtoNet', 'CovaMNet', 'RelationNet', 'DN4', 'ADM']:
		model = Fewshot_model(encoder_model=encoder_model, classifier_model=classifier_model, **kwargs)

	else:
		raise NotImplementedError('Model name [%s] is not recognized' % classifier_model)
	
	# init_weights(model, init_type=init_type)
	print_network(model)

	if use_gpu:
		model.cuda()

	if pretrained:
		model.load_state_dict(model_root)

	return model




class Fewshot_model(nn.Module):
	'''
		Define a few-shot learning model, which consists of an embedding module and a classifier moduel.
	'''
	def __init__(self, encoder_model='Conv64F', classifier_model='ProtoNet', class_num=64, way_num=5, shot_num=5, query_num=10, neighbor_k=3):
		super(Fewshot_model, self).__init__()
		self.encoder_model = encoder_model
		self.classifier_model = classifier_model
		self.way_num = way_num
		self.shot_num = shot_num
		self.query_num = query_num
		self.neighbor_k = neighbor_k
		self.loss_type = 'softmax'

		if   encoder_model == 'Conv64F':
			self.feature_dim = 64
		elif encoder_model == 'Conv64F_Li':
			self.feature_dim = 64
		elif encoder_model in ['ResNet10', 'ResNet18', 'ResNet34']:
			self.feature_dim = 512
		elif encoder_model in ['ResNet12', 'SeResNet12', 'ResNet12_Li', 'SeResNet12_Li']:
			self.feature_dim = 640
		elif encoder_model in ['ResNet50', 'ResNet101']:
			self.feature_dim = 2048
		
		encoder_module    = encoder_dict[self.encoder_model]
		classifier_module = classifier_dict[self.classifier_model]

		self.features   = encoder_module()
		self.classifier = classifier_module(way_num=self.way_num, shot_num=self.shot_num, neighbor_k=self.neighbor_k)
	

		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)


	def forward(self, input1, input2, is_feature=False):
		
		# pdb.set_trace()
		x1 = self.features(input1)      # query:       75 * 64 * 21 * 21   
		x2 = self.features(input2)      # support set: 25 * 64 * 21 * 21  
		
		out = self.classifier(x1, x2)

		if is_feature:
			return x1, x2, out
		else:
			return out


	def construct_episodes(self, aug_support, aug_s_targets):
		''' 
			Construct one episode each time.
			aug_support:    100 * 64 * 5 * 5  
			aug_s_targets:  100   
		'''

		target_unique = torch.unique(aug_s_targets, sorted=True)

		support = []
		query = []
		query_target = []
		for item in range(target_unique.size(0)):

			current_target = target_unique[item].unsqueeze(0)
			current_target_index = (aug_s_targets==current_target).nonzero()
			index = np.arange(current_target_index.size(0))
			random.shuffle(index)

			s_index = index[0:self.shot_num]
			q_index = index[self.shot_num:self.shot_num+self.query_num]
			support_index = current_target_index[s_index]
			query_index = current_target_index[q_index]

			support.append(support_index)
			query.append(query_index)
			query_target.append(current_target.expand(query_index.size(0), 1))

		support = torch.cat(support, 0).squeeze(1)
		query = torch.cat(query, 0).squeeze(1)
		query_target = torch.cat(query_target, 0).squeeze(1)


		return support, query, query_target



	def set_forward_adaptation(self, query, support, aug_support, aug_s_target):
		'''
			query:          75 * 64 * 5 * 5
			support:        25 * 64 * 5 * 5
			aug_support:    100 * 64 * 5 * 5  
			aug_s_target:   100  
		'''

		# define a new classifier
		# classifier_module = classifier_dict[self.classifier_model]
		# new_classifier = classifier_module(way_num=self.way_num, shot_num=self.shot_num, neighbor_k=self.neighbor_k).cuda()
		# new_classifier.load_state_dict(self.classifier)


		new_classifier = copy.deepcopy(self.classifier)
		for p in new_classifier.parameters():
			p.requires_grad = True
		# params = new_classifier.state_dict() 
		# pdb.set_trace()

		# define loss function (criterion) and optimizer
		criterion = nn.CrossEntropyLoss().cuda()
		# set_optimizer = torch.optim.SGD(new_classifier.parameters(), lr=0.1, momentum=0.9, dampening=0.9, weight_decay=0.001)
		set_optimizer = torch.optim.Adam(new_classifier.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=0.001)


		# ========= Fine tune ========#
		new_classifier.train()
		episode_num = 200
		for i in range(episode_num):

			# random sampling an episode based on the augmented support set
			support_index, query_index, new_query_target = self.construct_episodes(aug_support, aug_s_target)
			
			temp_support = aug_support[support_index]
			temp_query = aug_support[query_index]

			# Calculate the output
			temp_output = new_classifier(temp_query, temp_support)
			temp_loss = criterion(temp_output, new_query_target)

			# Compute gradients and do SGD step
			set_optimizer.zero_grad()
			temp_loss.backward()
			set_optimizer.step()
			# print('Iteration: {0} Loss: {loss:.3f}\t'.format(i, loss=temp_loss))

		new_classifier.eval()
		scores = new_classifier(query, support)
		return scores


	def set_forward_adaptation2(self, query, support, aug_support, aug_s_target):
		'''
			query:          75 * 64 * 5 * 5
			support:        25 * 64 * 5 * 5
			aug_support:    100 * 64 * 5 * 5  
			aug_s_target:   100   
		'''

		if self.loss_type == 'softmax':
			avgpool = nn.AdaptiveAvgPool2d(1)
			new_classifier = nn.Linear(self.feature_dim, self.way_num).cuda()
		elif self.loss_type == 'dist':       
			avgpool = nn.AdaptiveAvgPool2d(1) 
			new_classifier = classifier.distLinear(self.feature_dim, self.way_num).cuda()


		# define loss function (criterion) and optimizer
		criterion = nn.CrossEntropyLoss().cuda()
		# set_optimizer = torch.optim.SGD(new_classifier.parameters(), lr=0.1, momentum=0.9, dampening=0.9, weight_decay=0.001)
		set_optimizer = torch.optim.Adam(new_classifier.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=0.001)


		# ========= Fine tune ========#
		new_classifier.train()
		batch_size = 10
		support_size = aug_support.size(0)
		for epoch in range(20):
			rand_id = np.random.permutation(support_size)
			for i in range(0, support_size, batch_size):

				# random sampling a mini-batch
				selected_id = torch.from_numpy(rand_id[i: min(i+batch_size, support_size)]).cuda()
				z_batch = aug_support[selected_id]
				y_batch = aug_s_target[selected_id] 

				# Calculate the output
				z_batch = avgpool(z_batch).squeeze(3).squeeze(2)
				scores = new_classifier(z_batch)
				loss = criterion(scores, y_batch)

				# Compute gradients and do SGD step
				set_optimizer.zero_grad()
				loss.backward()
				set_optimizer.step()

		new_classifier.eval()
		query = avgpool(query).squeeze(3).squeeze(2)
		scores = new_classifier(query)

		return scores


	def set_forward_adaptation3(self, query, query_target, support, aug_support, aug_s_target):
		'''
			query:          75 * 64 * 5 * 5
			support:        25 * 64 * 5 * 5
			aug_support:    100 * 64 * 5 * 5  
			aug_s_target:   100   
		'''
	
		avgpool = nn.AdaptiveAvgPool2d(1)
		new_classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
									 multi_class='multinomial')

  
		aug_support = avgpool(aug_support).squeeze(3).squeeze(2)
		aug_support = aug_support.detach().cpu().numpy()
		aug_s_target = aug_s_target.detach().cpu().numpy()

		new_classifier.fit(aug_support, aug_s_target)

		query = avgpool(query).squeeze(3).squeeze(2)
		query = query.detach().cpu().numpy()
		query_target = query_target.detach().cpu().numpy()
		query_pred = new_classifier.predict(query)

		acc = metrics.accuracy_score(query_target, query_pred)

		return 100*acc




class General_model(nn.Module):
	'''
		Define a general image classification model, which consists of an embedding module and a classifier moduel.
	'''
	def __init__(self, encoder_model='Conv64F', classifier_model='Baseline', class_num=64, way_num=5, shot_num=5, query_num=10, neighbor_k=3):
		super(General_model, self).__init__()
		self.class_num = class_num
		self.way_num = way_num
		self.shot_num = shot_num

		if   encoder_model == 'Conv64F':
			self.feature_dim = 64
		elif encoder_model == 'Conv64F_Li':
			self.feature_dim = 64
		elif encoder_model in ['ResNet10', 'ResNet18', 'ResNet34']:
			self.feature_dim = 512
		elif encoder_model in ['ResNet12', 'SeResNet12', 'ResNet12_Li', 'SeResNet12_Li']:
			self.feature_dim = 640
		elif encoder_model in ['ResNet50', 'ResNet101']:
			self.feature_dim = 2048

		if   classifier_model == 'Baseline':
			self.loss_type = 'softmax'
		elif classifier_model == 'Baseline_plus':
			self.loss_type = 'dist'
		elif classifier_model == 'LR':
			self.loss_type = 'LR'

		encoder_module    = encoder_dict[encoder_model]
		classifier_module = classifier_dict[classifier_model]

		self.features   = encoder_module()
		self.classifier = classifier_module(class_num=self.class_num, feature_dim=self.feature_dim, loss_type=self.loss_type)
		self.rot_classifier = nn.Linear(self.class_num, 4)


	def forward(self, x, is_feature=False, rot=False):
		# pdb.set_trace()
		x   = self.features(x)
		out = self.classifier(x)

		if rot:
			x_rot = self.rot_classifier(out)
			return out, x_rot

		if is_feature:
			return x
		else:
			return out


	def set_forward_adaptation(self, query, support, aug_support, aug_s_target):
		'''
			query:          75 * 64 * 5 * 5
			support:        25 * 64 * 5 * 5
			aug_support:    100 * 64 * 5 * 5  
			aug_s_target:   100   

			(1) "A Closer Look at Few-shot Classification. ICLR 2019."
		'''

		if self.loss_type == 'softmax':
			avgpool = nn.AdaptiveAvgPool2d(1)
			new_classifier = nn.Linear(self.feature_dim, self.way_num).cuda()
		elif self.loss_type == 'dist':       
			avgpool = nn.AdaptiveAvgPool2d(1) 
			new_classifier = classifier.distLinear(self.feature_dim, self.way_num).cuda()
	  

		# define loss function (criterion) and optimizer
		criterion = nn.CrossEntropyLoss().cuda()
		# set_optimizer = torch.optim.SGD(new_classifier.parameters(), lr=0.1, momentum=0.9, dampening=0.9, weight_decay=0.001)
		set_optimizer = torch.optim.Adam(new_classifier.parameters(), lr=0.01, betas=(0.5, 0.9), weight_decay=0.001)


		# ========= Fine tune ========#
		new_classifier.train()
		batch_size = 10
		support_size = aug_support.size(0)
		for epoch in range(20):
			rand_id = np.random.permutation(support_size)
			for i in range(0, support_size, batch_size):

				# random sampling a mini-batch
				selected_id = torch.from_numpy(rand_id[i: min(i+batch_size, support_size)]).cuda()
				z_batch = aug_support[selected_id]
				y_batch = aug_s_target[selected_id] 

				# Calculate the output
				z_batch = avgpool(z_batch).squeeze(3).squeeze(2)
				scores = new_classifier(z_batch)
				loss = criterion(scores, y_batch)

				# Compute gradients and do SGD step
				set_optimizer.zero_grad()
				loss.backward()
				set_optimizer.step()

		new_classifier.eval()
		query = avgpool(query).squeeze(3).squeeze(2)
		scores = new_classifier(query)

		return scores


	def set_forward_adaptation2(self, query, query_target, support, aug_support, aug_s_target):
		'''
			query:          75 * 64 * 5 * 5
			support:        25 * 64 * 5 * 5
			aug_support:    100 * 64 * 5 * 5  
			aug_s_target:   100   
		    (2) "Rethinking Few-shot Image Classification: a Good Embedding Is All Your Need? ECCV 2020"
		     LR Classifier is used.
		'''
	
		avgpool = nn.AdaptiveAvgPool2d(1)
		new_classifier = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')

  
		aug_support = avgpool(aug_support).squeeze(3).squeeze(2)

		# L2 Normalization
		aug_support_norm = torch.norm(aug_support, 2, 1, True)                        # 75 * 1
		aug_support = aug_support/aug_support_norm

		aug_support = aug_support.detach().cpu().numpy()
		aug_s_target = aug_s_target.detach().cpu().numpy()

		new_classifier.fit(aug_support, aug_s_target)

		query = avgpool(query).squeeze(3).squeeze(2)

		# L2 Normalization
		query_norm = torch.norm(query, 2, 1, True)                                   # 75 * 1
		query = query/query_norm     
		
		query = query.detach().cpu().numpy()
		query_target = query_target.detach().cpu().numpy()
		query_pred = new_classifier.predict(query)

		acc = metrics.accuracy_score(query_target, query_pred)

		return 100*acc


class Model_with_reused_Encoder(nn.Module):
	'''
		Construct a new few-shot model by reusing a pre-trained embedding module.
	'''
	def __init__(self, pre_trained_model, new_classifier='ProtoNet', way_num=5, shot_num=5, neighbor_k=3):
		super(Model_with_reused_Encoder, self).__init__()
		self.way_num = way_num
		self.shot_num = shot_num
		self.neighbor_k = neighbor_k
		self.model = pre_trained_model

		# Only use the features module
		self.features = nn.Sequential(
			*list(self.model.features.children())
			)

		classifier_module = classifier_dict[new_classifier]
		self.classifier = classifier_module(way_num=self.way_num, shot_num=self.shot_num, neighbor_k=self.neighbor_k)


	def forward(self, input1, input2):
		
		# pdb.set_trace()
		x1 = self.features(input1)
		x2 = self.features(input2)
		out = self.classifier(x1, x2)

		return out


class Feature_Extractor_with_reused_Encoder(nn.Module):
	'''
		Construct a new few-shot model by reusing a pre-trained embedding module.
	'''
	def __init__(self, pre_trained_model):
		super(Feature_Extractor_with_reused_Encoder, self).__init__()
		self.model = pre_trained_model

		# Only use the features module
		self.features = nn.Sequential(
			*list(self.model.features.children())
			)

	def forward(self, input):
		
		# pdb.set_trace()
		out = self.features(input)

		return out

