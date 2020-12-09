##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Wenbin Li
## Date: Dec. 16 2018
##
## Divide data into train/val/test in a csv version
## Output: train.csv, val.csv, test.csv 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import csv
import numpy as np
import random
from PIL import Image
import pdb


data_dir = '/FewShot/Datasets/Stanford_cars'                # the path of the download dataset
save_dir = '/FewShot/Datasets/Stanford_cars/For_FewShot'    # the saving path of the divided dataset


if not os.path.exists(os.path.join(save_dir, 'images')):
	os.makedirs(os.path.join(save_dir, 'images'))

images_dir = os.path.join(data_dir, 'images')
train_class_num = 130
val_class_num =  17
test_class_num = 49



# get all the dog classes
classes_list = [class_name for class_name in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, class_name))]


# divide the train/val/test set
random.seed(196)
train_list = random.sample(classes_list, train_class_num)
remain_list = [rem for rem in classes_list if rem not in train_list]
val_list = random.sample(remain_list, val_class_num)
test_list = [rem for rem in remain_list if rem not in val_list]


# save data into csv file----- Train
train_data = []
for class_name in train_list:
	images = [[i, class_name] for i in os.listdir(os.path.join(images_dir, class_name))]
	train_data.extend(images)
	print('Train----%s' %class_name)

	# read images and store these images
	img_paths = [os.path.join(images_dir, class_name, i) for i in os.listdir(os.path.join(images_dir, class_name))]
	for index, img_path in enumerate(img_paths):
		img = Image.open(img_path)
		img = img.convert('RGB')
		img.save(os.path.join(save_dir, 'images', images[index][0]), quality=100)


with open(os.path.join(save_dir, 'train.csv'), 'w') as csvfile:
	writer = csv.writer(csvfile)

	writer.writerow(['filename', 'label'])
	writer.writerows(train_data)




# save data into csv file----- Val
val_data = []
for class_name in val_list:
	images = [[i, class_name] for i in os.listdir(os.path.join(images_dir, class_name))]
	val_data.extend(images)
	print('Val----%s' %class_name)

	# read images and store these images
	img_paths = [os.path.join(images_dir, class_name, i) for i in os.listdir(os.path.join(images_dir, class_name))]
	for index, img_path in enumerate(img_paths):
		img = Image.open(img_path)
		img = img.convert('RGB')
		img.save(os.path.join(save_dir, 'images', images[index][0]), quality=100)

with open(os.path.join(save_dir, 'val.csv'), 'w') as csvfile:
	writer = csv.writer(csvfile)

	writer.writerow(['filename', 'label'])
	writer.writerows(val_data)




# save data into csv file----- Test
test_data = []
for class_name in test_list:
	images = [[i, class_name] for i in os.listdir(os.path.join(images_dir, class_name))]
	test_data.extend(images)
	print('Test----%s' %class_name)

	# read images and store these images
	img_paths = [os.path.join(images_dir, class_name, i) for i in os.listdir(os.path.join(images_dir, class_name))]
	for index, img_path in enumerate(img_paths):
		img = Image.open(img_path)
		img = img.convert('RGB')
		img.save(os.path.join(save_dir, 'images', images[index][0]), quality=100)


with open(os.path.join(save_dir, 'test.csv'), 'w') as csvfile:
	writer = csv.writer(csvfile)

	writer.writerow(['filename', 'label'])
	writer.writerows(test_data)
