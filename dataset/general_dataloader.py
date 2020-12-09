import os
import os.path as path
import json
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import pdb
import csv
import pickle
import sys
sys.dont_write_bytecode = True
#torch.multiprocessing.set_sharing_strategy('file_system')


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def RGB_loader(path):
	return Image.open(path).convert('RGB')


def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def gray_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('P')


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)


def find_classes(dir):
	classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
	classes.sort()
	class_to_idx = {classes[i]: i for i in range(len(classes))}

	return classes, class_to_idx


def load_csv2dict(csv_path):
	class_img_dict = {}
	with open(csv_path) as csv_file:
		csv_context = csv.reader(csv_file, delimiter=',')
		for line in csv_context:
			if csv_context.line_num == 1:
				continue
			img_name, img_class = line

			if img_class in class_img_dict:
				class_img_dict[img_class].append(img_name)
			else:
				class_img_dict[img_class] = []
				class_img_dict[img_class].append(img_name)

	csv_file.close()
	class_list = list(class_img_dict.keys())
	class_list.sort()
	class_to_idx = {class_list[i]: i for i in range(len(class_list))}

	return class_img_dict, class_list, class_to_idx


def data_split(data_dir, class_img_dict, class_list, class_to_idx, mode):
	'''
		Split the auxiliary dataset into train set and val set for general classification.
	'''
	Train_data_list = []
	Val_data_list   = []
	Test_data_list  = []

	random.seed(100) # set s seed
	for index, class_item in enumerate(class_list):
		temp_imgs = class_img_dict[class_item]
		target = class_to_idx[class_item]

		temp_list = []
		for j in range(len(temp_imgs)):
			data_file = {
				"img": temp_imgs[j],
				"target": target
				}
			temp_list.append(data_file)

		# pdb.set_trace()
		# divide the train/val/test set, i.e., 500/50/50  
		train_part  = random.sample(temp_list, 550)
		remain_part = [rem for rem in temp_list if rem not in train_part]
		val_part    = random.sample(remain_part, 30)
		test_part   = [te for te in remain_part if te not in val_part]
	

		# store the dataset
		Train_data_list.extend(train_part)
		Val_data_list.extend(val_part)
		Test_data_list.extend(test_part)


	# save the train_split
	train_save_path = os.path.join(data_dir, 'train_part_list.pkl')
	train_output = open(train_save_path, 'wb')
	pickle.dump(Train_data_list, train_output)
	train_output.close()


	# save the val_split
	val_save_path = os.path.join(data_dir, 'val_part_list.pkl')
	val_output = open(val_save_path, 'wb')
	pickle.dump(Val_data_list, val_output)
	val_output.close()


	# save the val_split
	test_save_path = os.path.join(data_dir, 'test_part_list.pkl')
	test_output = open(test_save_path, 'wb')
	pickle.dump(Test_data_list, test_output)
	test_output.close()


	if mode == 'train':
		return Train_data_list

	elif mode == 'val':
		return Val_data_list

	elif mode == 'test':
		return Test_data_list




def read_dataset(data_dir):
	'''
		Read the train/val splits from the disk.
	'''
	data_list = pickle.load(open(data_dir, 'rb'))

	return data_list


def episode_sampling(data_dir, class_list, class_img_dict, episode_num, way_num=5, shot_num=5, query_num=15):
	'''
		Random constructing episodes from the dataset
		episode_num: the total number of episodes 
	'''
	data_list = []
	for e in range(episode_num):

		# construct each episode
		episode = []
		temp_list = random.sample(class_list, way_num)
		label_num = -1 

		for item in temp_list:
			label_num += 1
			imgs_set = class_img_dict[item]
			support_imgs = random.sample(imgs_set, shot_num)
			query_imgs = [val for val in imgs_set if val not in support_imgs]

			if query_num < len(query_imgs):
				query_imgs = random.sample(query_imgs, query_num)

			# the dir of support set
			query_dir = [path.join(data_dir, 'images', i) for i in query_imgs]
			support_dir = [path.join(data_dir, 'images', i) for i in support_imgs]

			data_files = {
				"query_img": query_dir,
				"support_set": support_dir,
				"target": label_num
			}
			episode.append(data_files)
		data_list.append(episode)

	return data_list



#================ For the general image classification tasks ================#

class GeneralDataSet(object):
	'''
		Prepare the datasets for training, validation and test.
	'''
	def __init__(self, opt, transform=None, mode='train', loader=RGB_loader):
		super(GeneralDataSet, self).__init__()

		self.mode = mode
		self.transform = transform
		self.loader = loader
		self.way_num = opt.way_num
		self.shot_num = opt.shot_num
		self.query_num = opt.query_num
		self.data_dir = opt.dataset_dir

		assert (mode in ['train', 'val', 'test'])

		# print('mode {0} --- Loading dataset'.format(mode))
		if mode == 'train':
			csv_path    = os.path.join( self.data_dir, 'train.csv')
			data_path = os.path.join(self.data_dir, 'train_part_list.pkl')

		elif mode == 'val': # We still use the train.csv
			csv_path    = os.path.join( self.data_dir, 'train.csv')
			data_path = os.path.join(self.data_dir, 'val_part_list.pkl')

		elif mode == 'test': 
			csv_path    = os.path.join( self.data_dir, 'train.csv')
			data_path = os.path.join(self.data_dir, 'test_part_list.pkl')

		else:
			raise ValueError('mode ought to be in [train/test/val]')


		# Check whether the splits have been saved in the disk
		if os.path.exists(data_path):
			self.data_list = read_dataset(data_path)
		else:
			class_img_dict, class_list, class_to_idx = load_csv2dict(csv_path)
			self.data_list = data_split(self.data_dir, class_img_dict, class_list, class_to_idx, mode)

		print('Loading dataset -- mode {0}: {1}'.format(mode, len(self.data_list)))


	def __len__(self):
		return len(self.data_list)


	def __getitem__(self, index):
		'''
			Load an image for training and validation.          
		'''
		data_file = self.data_list[index]
		img_path = os.path.join(self.data_dir, 'images', data_file['img'])
		img = self.loader(img_path)
		target = data_file['target']

		# Normalization
		if self.transform is not None:
			img = self.transform(img)

		return (img, target)



#================ For the Few-shot image classification tasks ================#

class FewShotDataSet_Wenbin(object):
	'''
		Prepare the datasets of episodes for training, validation and test.
	'''
	def __init__(self, opt, transform=None, support_transform=None, mode='train', loader=RGB_loader):
		super(FewShotDataSet_Wenbin, self).__init__()

		self.mode = mode
		self.transform = transform
		self.support_transform = support_transform
		self.loader = loader
		self.way_num = opt.way_num
		self.shot_num = opt.shot_num
		self.query_num = opt.query_num
		self.data_dir = opt.dataset_dir
		self.test_aug = opt.test_aug
		self.augmented_shot_num = opt.aug_shot_num


		assert (mode in ['train', 'val', 'test'])

		
		if mode == 'train':
			csv_path    = os.path.join( self.data_dir, 'train.csv')
			self.episode_num = opt.episode_train_num
		
		elif mode == 'val':
			csv_path    = os.path.join( self.data_dir, 'val.csv')
			self.episode_num = opt.episode_val_num

		elif mode == 'test': # This part is not available in this version
			csv_path    = os.path.join( self.data_dir, 'test.csv')
			self.episode_num = opt.episode_test_num

		else:
			raise ValueError('mode ought to be in [train/test/val]')


		# Construct the few-shot tasks (episodes)
		class_img_dict, class_list, class_to_idx = load_csv2dict(csv_path)
		self.data_list = episode_sampling(self.data_dir, class_list, class_img_dict, 
			self.episode_num, self.way_num, self.shot_num, self.query_num)

		print('Loading dataset -- mode {0}: {1} (Few-shot)'.format(mode, len(self.data_list)))


	def __len__(self):
		return len(self.data_list)


	def __getitem__(self, index):
		'''
			Load an episode for training and validation.          
		'''
		episode_files = self.data_list[index]

		query_images = []
		query_targets = []
		support_images = []
		support_targets = []
		augmented_support_images = []
		augmented_support_targets = []

		if self.test_aug and self.mode == 'test':
		
			for i in range(len(episode_files)):
				data_files = episode_files[i]

				# load query images
				query_dir = data_files['query_img']

				for j in range(len(query_dir)):
					temp_img = self.loader(query_dir[j])

					# Normalization
					if self.transform is not None:
						temp_img = self.transform(temp_img)
					query_images.append(temp_img)


				# load support images
				temp_support = []
				temp_augmented_support = []
				support_dir = data_files['support_set']
				for j in range(len(support_dir)): 
					PIL_img = self.loader(support_dir[j])

					# Normalization
					if self.transform is not None:
						temp_img = self.transform(PIL_img)
						temp_support.append(temp_img.unsqueeze(0))

					if self.support_transform is not None:
						for _ in range(self.augmented_shot_num):
							temp_img = self.support_transform(PIL_img)
							temp_augmented_support.append(temp_img.unsqueeze(0))

				temp_support = torch.cat(temp_support, 0)
				support_images.append(temp_support)
				temp_augmented_support = torch.cat(temp_augmented_support, 0)
				augmented_support_images.append(temp_augmented_support)


				# read the label
				target = data_files['target']
				query_targets.extend(np.tile(target, len(query_dir)))
				support_targets.extend(np.tile(target, len(support_dir)))
				augmented_support_targets.extend(np.tile(target, len(support_dir) * self.augmented_shot_num))

			 
			return (query_images, query_targets, support_images, support_targets, augmented_support_images, augmented_support_targets)

		else:

			for i in range(len(episode_files)):
				data_files = episode_files[i]

				# load query images
				query_dir = data_files['query_img']

				for j in range(len(query_dir)):
					temp_img = self.loader(query_dir[j])

					# Normalization
					if self.transform is not None:
						temp_img = self.transform(temp_img)
					query_images.append(temp_img)


				# load support images
				temp_support = []
				support_dir = data_files['support_set']
				for j in range(len(support_dir)): 
					PIL_img = self.loader(support_dir[j])

					# Normalization
					if self.transform is not None:
						temp_img = self.transform(PIL_img)
						temp_support.append(temp_img.unsqueeze(0))

				temp_support = torch.cat(temp_support, 0)
				support_images.append(temp_support)

				# read the label
				target = data_files['target']
				query_targets.extend(np.tile(target, len(query_dir)))
				support_targets.extend(np.tile(target, len(support_dir)))
			 
			return (query_images, query_targets, support_images, support_targets)





def get_dataloader(opt, modes):
	'''
		Obtain the data loader for training/val/test.
	'''
	loaders = []
	mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
	std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
	for mode in modes:

		if opt.train_aug and opt.imageSize == 224 and mode == 'train':

			transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomCrop(opt.imageSize),
				transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				# transforms.Normalize(mean=mean, std=std)
			])

		elif opt.train_aug and opt.imageSize == 84 and mode == 'train':

			transform = transforms.Compose([
				transforms.Resize((92, 92)),
				transforms.RandomCrop(opt.imageSize),
				transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				# transforms.Normalize(mean=mean, std=std)
			])

		else:

			transform = transforms.Compose([
				transforms.Resize((opt.imageSize, opt.imageSize)),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				# transforms.Normalize(mean=mean, std=std)
			])


		dataset = GeneralDataSet(opt, transform, mode)


		if mode == 'train':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.batch_size, shuffle=True,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		elif mode == 'val':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.batch_size, shuffle=True,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		elif mode == 'test':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.batch_size, shuffle=False,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		else:
			raise ValueError('Mode ought to be in [train, val, test]')

		loaders.append(loader)

	return loaders




def get_Fewshot_dataloader(opt, modes):
	'''
		Obtain the data loader for training/val/test.
	'''
	loaders = []
	mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
	std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
	for mode in modes:

		if opt.train_aug and opt.imageSize == 224 and mode == 'train':

			transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomCrop(opt.imageSize),
				transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				# transforms.Normalize(mean=mean, std=std)
			])

		elif opt.train_aug and opt.imageSize == 84 and mode == 'train':

			transform = transforms.Compose([
				transforms.Resize((92, 92)),
				transforms.RandomCrop(opt.imageSize),
				transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				# transforms.Normalize(mean=mean, std=std)
			])

		else:

			transform = transforms.Compose([
				transforms.Resize((opt.imageSize, opt.imageSize)),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				# transforms.Normalize(mean=mean, std=std)
			])


		if opt.imageSize == 224:
			""" Use to generate additional support set"""
			support_transform = transforms.Compose([
					transforms.Resize((256, 256)),
					transforms.RandomCrop(opt.imageSize),
					transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
					# transforms.Normalize(mean=mean, std=std)
				])
		else:
			support_transform = transforms.Compose([
					transforms.Resize((92, 92)),
					transforms.RandomCrop(opt.imageSize),
					transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
					# transforms.Normalize(mean=mean, std=std)
				])

	
		dataset = FewShotDataSet_Wenbin(opt, transform, support_transform, mode)


		if mode == 'train':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.episodeSize, shuffle=True,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		elif mode == 'val':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.episodeSize, shuffle=True,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		elif mode == 'test':
			loader = torch.utils.data.DataLoader(
				dataset, batch_size=opt.testepisodeSize, shuffle=False,
				num_workers=int(opt.workers), drop_last=True, pin_memory=True)
		else:
			raise ValueError('Mode ought to be in [train, val, test]')

		loaders.append(loader)

	return loaders