import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import utils
import pdb
from abc import abstractmethod
from torch.nn.utils.weight_norm import WeightNorm


# =========================== Fine tune a new classifier model =========================== #
class Finetune_Classifier(nn.Module):
	'''
		Construct a new classifier module for a new task.
	'''
	def __init__(self, class_num=5, feature_dim=64, loss_type = 'softmax'):
		super(Finetune_Classifier, self).__init__()
		self.class_num = class_num
		self.feature_dim = feature_dim
		self.loss_type = loss_type       #'softmax' or #'dist'

		if loss_type == 'softmax':  # Baseline
			self.avgpool = nn.AdaptiveAvgPool2d(1)
			self.classifier = nn.Linear(self.feature_dim, self.class_num)

		elif loss_type == 'dist':   # Baseline ++
			self.avgpool = nn.AdaptiveAvgPool2d(1)
			self.classifier = distLinear(self.feature_dim, self.class_num)


	def forward(self, x):
	
		x = self.avgpool(x).squeeze(3).squeeze(2)	
		# x = x.view(x.size(0), -1)             
		scores = self.classifier(x)

		return scores




# =========================== General classification method: Baseline =========================== #
class Baseline_Metric(nn.Module):
	'''
		Classifier module of the general image classification task.
		1. Baseline
		2. Baseline ++ 
		   Note that Both of them are parametric classifiers.
		   (1) "A Closer Look at Few-shot Classification. ICLR 2019."
		   (2) "Rethinking Few-shot Image Classification: a Good Embedding Is All Your Need? ECCV 2020"
		   (3) "Self-supervised Knowledge Distillation for Few-shot Learning. arXiv 2020"

		Note that all the above paper will first prr-train a classifier but use different strategies at the test stage. 
	'''
	def __init__(self, class_num=64, feature_dim=64, loss_type='softmax'):
		super(Baseline_Metric, self).__init__()
		self.class_num = class_num
		self.feature_dim = feature_dim
		self.loss_type = loss_type       #'softmax' or #'dist'

		if loss_type == 'softmax':  # Baseline
			self.avgpool = nn.AdaptiveAvgPool2d(1)
			self.classifier = nn.Linear(self.feature_dim, self.class_num)
			

		elif loss_type == 'dist':   # Baseline ++
			self.avgpool = nn.AdaptiveAvgPool2d(1)
			self.classifier = distLinear(self.feature_dim, self.class_num)
			

	def forward(self, x):
	
		x = self.avgpool(x).squeeze(3).squeeze(2)	             
		scores = self.classifier(x)

		return scores


class distLinear(nn.Module):
	'''
		Coming from "A Closer Look at Few-shot Classification. ICLR 2019."
		https://github.com/wyharveychen/CloserLookFewShot.git
	'''
	def __init__(self, indim, outdim):
		super(distLinear, self).__init__()
		self.L = nn.Linear( indim, outdim, bias = False)
		self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
		if self.class_wise_learnable_norm:      
			WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

		if outdim <=200:
			self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
		else:
			self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

	def forward(self, x):
		x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
		x_normalized = x.div(x_norm+ 0.00001)
		if not self.class_wise_learnable_norm:
			L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
			self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
		cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm
		scores = self.scale_factor* (cos_dist) 

		return scores



# =========================== Few-shot learning method: ProtoNet =========================== #
class Prototype_Metric(nn.Module):
	'''
		The classifier module of ProtoNet by using the mean prototype and Euclidean distance,
		which is also Non-parametric.
		"Prototypical networks for few-shot learning. NeurIPS 2017."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(Prototype_Metric, self).__init__()
		self.way_num = way_num
		self.avgpool = nn.AdaptiveAvgPool2d(1)


	# Calculate the Euclidean distance between the query and the mean prototype of the support class.
	def cal_EuclideanDis(self, input1, input2):
		'''
		 input1 (query images): 75 * 64 * 5 * 5
		 input2 (support set):  25 * 64 * 5 * 5
		'''
	
		# input1---query images
		# query = input1.view(input1.size(0), -1)                                    # 75 * 1600     (Conv64F)
		query = self.avgpool(input1).squeeze(3).squeeze(2)                           # 75 * 64
		query = query.unsqueeze(1)                                                   # 75 * 1 * 1600 (Conv64F)
   

		# input2--support set
		input2 = self.avgpool(input2).squeeze(3).squeeze(2)                          # 25 * 64
		# input2 = input2.view(input2.size(0), -1)                                   # 25 * 1600     
		support_set = input2.contiguous().view(self.way_num, -1, input2.size(1))     # 5 * 5 * 1600    
		support_set = torch.mean(support_set, 1)                                     # 5 * 1600


		# Euclidean distances between a query set and a support set
		proto_dis = -torch.pow(query-support_set, 2).sum(2)                          # 75 * 5 
		

		return proto_dis


	def forward(self, x1, x2):

		proto_dis = self.cal_EuclideanDis(x1, x2)

		return proto_dis



# =========================== Few-shot learning method: RelationNet =========================== #
class LearnToCompare_Metric(nn.Module):
	'''
		Learn-to-compare classifier module for RelationNet, which is a parametric classifier.
		"Learning to Compare: Relation Network for Few-Shot Learning. CVPR 2018."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(LearnToCompare_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.shot_num = shot_num
		self.way_num = way_num


		# Relation Block of RelationNet
		self.RelationNetwork = nn.Sequential(
			nn.Conv2d(64*2,64,kernel_size=3,padding=0),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(64,64,kernel_size=3,padding=0),
			nn.BatchNorm2d(64, momentum=1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)

		self.fc1 = nn.Linear(64*3*3, 8)
		self.fc2 = nn.Linear(8,1)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)


	# Calculate the relationships between the query and support class via a Deep Metric Network.
	def cal_relationships(self, input1, input2):
		'''
		 input1 (query images):  75 * 64 * 21 * 21
		 input2 (support set):   25 * 64 * 21 * 21
		'''
		# pdb.set_trace()
		# input1---query images
		input1 = input1.unsqueeze(0).repeat(1*self.way_num,1,1,1,1)              # 5 * 75 * 64 * 21 * 21 (Conv64F_Li)
		query = input1.permute(1, 0, 2, 3, 4)                                    # 75 * 5 * 64 * 21 * 21 (Conv64F_Li)

		
		# input2--support set
		input2 = input2.contiguous().view(self.way_num, self.shot_num, input2.size(1), input2.size(2), input2.size(3))   # 5 * 5 * 64 * 21 * 21
		support = torch.sum(input2, 1).squeeze(1)                                                                        # 5 * 64 * 21 * 21
		support = support.unsqueeze(0).repeat(query.size(0), 1, 1, 1, 1)         # 75 * 5 * 64 * 21 * 21 (Conv64F_Li)


		# Concatenation 
		relation_pairs = torch.cat((support, query), 2).view(-1,input2.size(2)*2, input2.size(3), input2.size(4))
		out = self.RelationNetwork(relation_pairs)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1(out))
		relations = self.fc2(out).view(-1, self.way_num).cuda()

		return relations


	def forward(self, x1, x2):

		relations = self.cal_relationships(x1, x2)

		return relations




# =========================== Few-shot learning method: DN4 =========================== #
class ImgtoClass_Metric(nn.Module):
	'''
		Image-to-class classifier module for DN4, which is a Non-parametric classifier.
		"Revisiting local descriptor based image-to-class measure for few-shot learning. CVPR 2019."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.shot_num = shot_num


	# Calculate the Image-to-class similarity between the query and support class via k-NN.
	def cal_cosinesimilarity(self, input1, input2):
		'''
		 input1 (query images):  75 * 64 * 21 * 21
		 input2 (support set):   25 * 64 * 21 * 21
		'''

		# input1---query images
		input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1)         # 75 * 64 * 441 (Conv64F_Li)
		input1 = input1.permute(0, 2, 1)                                              # 75 * 441 * 64 (Conv64F_Li)

		
		# input2--support set
		input2 = input2.contiguous().view(input2.size(0), input2.size(1), -1)         # 25 * 64 * 441
		input2 = input2.permute(0, 2, 1)                                              # 25 * 441 * 64


		# L2 Normalization
		input1_norm = torch.norm(input1, 2, 2, True)                                  # 75 * 441 * 1
		query = input1/input1_norm                                                    # 75 * 441 * 64
		query = query.unsqueeze(1)                                                    # 75 * 1 * 441 *64


		input2_norm = torch.norm(input2, 2, 2, True)                                  # 25 * 441 * 1 
		support_set = input2/input2_norm                                              # 25 * 441 * 64
		support_set = support_set.contiguous().view(-1,
				self.shot_num*support_set.size(1), support_set.size(2))               # 5 * 2205 * 64    
		support_set = support_set.permute(0, 2, 1)                                    # 5 * 64 * 2205     


		# cosine similarity between a query set and a support set
		innerproduct_matrix = torch.matmul(query, support_set)                        # 75 * 5 * 441 * 2205


		# choose the top-k nearest neighbors
		topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 3)  # 75 * 5 * 441 * 3
		img2class_sim = torch.sum(torch.sum(topk_value, 3), 2)                        # 75 * 5 


		return img2class_sim


	def forward(self, x1, x2):

		img2class_sim = self.cal_cosinesimilarity(x1, x2)

		return img2class_sim



# =========================== Few-shot learning method: CovaMNet =========================== #
class Covariance_Metric(nn.Module):
	'''
		Covariance metric classifier module of CovaMNet.
		"Distribution Consistency based Covariance Metric Networks for Few-shot Learning. AAAI 2019."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(Covariance_Metric, self).__init__()
		self.shot_num = shot_num

		self.conv1d_layer = nn.Sequential(
			nn.LeakyReLU(0.2, True),
			nn.Dropout(),
			nn.Conv1d(1, 1, kernel_size=441, stride=441),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)


	def cal_covariance_Batch(self, feature):     
		'''
		   Calculate the Covariance Matrices based on the local descriptors for a mini-batch images.  
		   feature (support set): 25 * 64 * 21 * 21
		'''

		feature = feature.contiguous().view(feature.size(0), feature.size(1), -1)     # 25 * 64 * 441
		feature = feature.permute(0, 2, 1)                                            # 25 * 441 * 64
		feature = feature.contiguous().view(-1,
				self.shot_num*feature.size(1), feature.size(2))                       # 5 * 2205 * 64    
	

		n_local_descriptor = torch.tensor(feature.size(1)).cuda()
		feature_mean = torch.mean(feature, 1, True)                                   # 5 * 1 * 64
		feature = feature - feature_mean                                              # 5 * 2205 * 64
		cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)                  # 5 * 64 * 64
		cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)                    # 5 * 64 * 64

		return feature_mean, cov_matrix 


	# Calculate the covariance similarity between the query and support class.
	def cal_covasimilarity(self, input1, input2):
		'''
		 input1 (query images):                           75 * 64 * 21 * 21
		 input2 (Covariance matrix of the support set):   5 * 64 * 64 
		'''

		# pdb.set_trace()
		query = input1.contiguous().view(input1.size(0), input1.size(1), -1)          # 75 * 64 * 441
		query_mean = torch.mean(query, 2, True)                                       # 75 * 64 * 1
		query = query - query_mean 
		query = query.permute(0, 2, 1).unsqueeze(1)                                   # 75 * 1 * 441 * 64


		# query_sam = input[i]
		# query_sam = query_sam.view(C, -1)
		# mean_query = torch.mean(query_sam, 1, True)
		# query_sam = query_sam-mean_query

		# L2 Normalization
		# input_query_norm = torch.norm(input_query, 2, 2, True)                      # 75 * 441 * 1
		# query = input_query/input_query_norm                                        # 75 * 441 * 64
		# query = query.unsqueeze(1)                                                  # 75 * 1 * 441 *64
		


		# covariance similarity between a query sample and a support set
		product_matrix = torch.matmul(query, input2)                                    # 75 * 5 * 441 * 64
		product_matrix2 = torch.matmul(product_matrix, torch.transpose(query, 2, 3))    # 75 * 5 * 441 * 441
		product_matrix2 = product_matrix2.contiguous().view(-1, product_matrix2.size(2), product_matrix2.size(3))  # 375 * 441 * 441 
		Cova_Sim = [product_matrix2[i].diag() for i in range(product_matrix2.size(0))]

		Cova_Sim = torch.cat(Cova_Sim, 0)                                                # 375 * 441  
		Cova_Sim = Cova_Sim.contiguous().view(product_matrix.size(0), -1).unsqueeze(1)   # 75 * 1 * 2205 
		# Cova_Sim = torch.sum(Cova_Sim, 2)

		return Cova_Sim


	def forward(self, x1, x2):

		Mean_support, CovaMatrix = self.cal_covariance_Batch(x2)
		Cova_Sim = self.cal_covasimilarity(x1, CovaMatrix)
		scores = self.conv1d_layer(Cova_Sim).squeeze(1)

		return scores





#========================== Few-shot learning method: ADM ==========================#
class ADM_Metric(nn.Module):
	'''
		 Asymmetric Distribution measure layer.
		"Asymmetric Distribution Measure for Few-shot Learning. IJCAI 2020."
	'''
	def __init__(self, way_num=5, shot_num=1, neighbor_k=3):
		super(ADM_Metric, self).__init__()
		self.way_num = way_num
		self.neighbor_k = neighbor_k
		self.shot_num = shot_num

		self.Norm_layer = nn.BatchNorm1d(self.way_num*2, affine=True)
		self.FC_layer =  nn.Conv1d(1, 1, kernel_size=2, stride=1, dilation=5, bias=False)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight.data, 0.0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				init.normal_(m.weight.data, 1.0, 0.02)
				init.constant_(m.bias.data, 0.0)
				

	def cal_covariance_matrix_Batch(self, feature):   # feature: Batch * descriptor_num * 64
		n_local_descriptor = torch.tensor(feature.size(1)).cuda()
		feature_mean = torch.mean(feature, 1, True)   # Batch * 1 * 64
		feature = feature - feature_mean
		cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)
		cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)
		cov_matrix = cov_matrix + 0.01*torch.eye(cov_matrix.size(1)).cuda()

		return feature_mean, cov_matrix


	def cal_covariance_Batch(self, feature):     # feature: 25 * 64 * 21 * 21
		
		feature = feature.contiguous().view(feature.size(0), feature.size(1), -1)     # 25 * 64 * 441
		feature = feature.permute(0, 2, 1)                                            # 25 * 441 * 64
		

		n_local_descriptor = torch.tensor(feature.size(1)).cuda()
		feature_mean = torch.mean(feature, 1, True)   # Batch * 1 * 64
		feature = feature - feature_mean
		cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)
		cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)
		cov_matrix = cov_matrix + 0.01*torch.eye(cov_matrix.size(1)).cuda()

		return feature_mean, cov_matrix


	def support_remaining(self, S):   # S: 5 * 441 * 64

		S_new = []
		for ii in range(S.size(0)):
		
			indices = [j for j in range(S.size(0))]
			indices.pop(ii)
			indices = torch.tensor(indices).cuda()

			S_clone = S.clone()
			S_remain = torch.index_select(S_clone, 0, indices)           # 4 * 441 * 64
			S_remain = S_remain.contiguous().view(-1, S_remain.size(2))  # -1 * 64    
			S_new.append(S_remain.unsqueeze(0))

		S_new = torch.cat(S_new, 0)   # 5 * 1764 * 64 
		
		return S_new



	def KL_distance_Batch(self, mean1, cov1, mean2, cov2):
		'''
		   mean1: 75 * 1 * 64
		   cov1:  75 * 64 * 64
		   mean2: 5 * 1 * 64
		   cov2: 5 * 64 * 64
		'''
		
		cov2_inverse = torch.inverse(cov2)            # 5 * 64 * 64
		cov1_det = torch.det(cov1)                    # 75 
		cov2_det = torch.det(cov2)                    # 5 
		mean_diff = -(mean1 - mean2.squeeze(1))       # 75 * 5 * 64


		# Calculate the trace
		matrix_product = torch.matmul(cov1.unsqueeze(1), cov2_inverse)  # 75 * 5 * 64 * 64
		trace_dis = [torch.trace(matrix_product[j][i]).unsqueeze(0)  for j in range(matrix_product.size(0)) for i in range(matrix_product.size(1))]
		trace_dis = torch.cat(trace_dis, 0)
		trace_dis = trace_dis.view(matrix_product.size(0), matrix_product.size(1))   # 75 * 5 


		# Calcualte the Mahalanobis Distance
		maha_product = torch.matmul(mean_diff.unsqueeze(2), cov2_inverse) # 75 * 5 * 1 * 64
		maha_product = torch.matmul(maha_product, mean_diff.unsqueeze(3)) # 75 * 5 * 1 * 1
		maha_product = maha_product.squeeze(3) 
		maha_product = maha_product.squeeze(2)         # 75 * 5 


		matrix_det = torch.logdet(cov2) - torch.logdet(cov1).unsqueeze(1)
		KL_dis = trace_dis + maha_product + matrix_det - mean1.size(2)

		return KL_dis/2.



	# Calculate KL divergence Distance
	def cal_ADM_similarity(self, input1, input2):
		'''
		 input1 (query images):  25 * 64 * 21 * 21
		 input2 (support set):   25 * 64 * 21 * 21
		'''
		# pdb.set_trace()
		# Calculate the mean and covariance of the all the query images
		query_mean, query_cov = self.cal_covariance_Batch(input1)   # query_mean: 75 * 1 * 64  query_cov: 75 * 64 * 64
		input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1)   #  75 * 64 * 441
		input1 = input1.permute(0, 2, 1)                                        #  75 * 441 * 64
		

		# Calculate the mean and covariance of the support set
		input2 = input2.contiguous().view(input2.size(0), input2.size(1), -1)   #  25 * 64 * 441
		input2 = input2.permute(0, 2, 1)                                        #  25 * 441 * 64


		# L2 Normalization
		input1_norm = torch.norm(input1, 2, 2, True)  # 75 * 441 * 1
		input2_norm = torch.norm(input2, 2, 2, True)  # 25 * 441 * 1


		support_set = input2.contiguous().view(-1, 
			self.shot_num*input2.size(1), input2.size(2))                       #  5 * (5*441) * 64    
		s_mean, s_cov = self.cal_covariance_matrix_Batch(support_set)           #  s_mean: 5 * 1 * 64  s_cov: 5 * 64 * 64


		# Find the remaining support set
		support_set_remain = self.support_remaining(support_set)
		s_remain_mean, s_remain_cov = self.cal_covariance_matrix_Batch(support_set_remain) # s_remain_mean: 5 * 1 * 64  s_remain_cov: 5 * 64 * 64


		# Calculate the Wasserstein Distance
		kl_dis = -self.KL_distance_Batch(query_mean, query_cov, s_mean, s_cov)  # 75 * 5


		# Calculate the Image-to-Class Similarity
		query_norm = input1/input1_norm      # 75 * 441 * 64
		support_norm = input2/input2_norm    # 25 * 441 * 64
		assert(torch.min(input1_norm)>0)
		assert(torch.min(input2_norm)>0)


		support_norm = support_norm.contiguous().view(-1, 
			self.shot_num*support_norm.size(1), support_norm.size(2))       # 5 * 2205 * 64   

		# cosine similarity between a query set and a support set
		innerproduct_matrix = torch.matmul(query_norm.unsqueeze(1), support_norm.permute(0, 2, 1))    # 75 * 5 * 441 * 2205


		# choose the top-k nearest neighbors
		topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 3)    # 75 * 5 * 441 * 1
		inner_sim = torch.sum(torch.sum(topk_value, 3), 2)                              # 75 * 5
	
	
		# Using FC layer to combine two parts ---- The original 
		ADM_sim_soft = torch.cat((kl_dis, inner_sim), 1)#.unsqueeze(1)
		ADM_sim_soft = self.Norm_layer(ADM_sim_soft).unsqueeze(1)
		ADM_sim_soft = self.FC_layer(ADM_sim_soft).squeeze(1)

		
		return ADM_sim_soft



	def forward(self, x1, x2):


		ADM_sim = self.cal_ADM_similarity(x1, x2)

		return ADM_sim
