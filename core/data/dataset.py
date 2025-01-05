# -*- coding: utf-8 -*-
import csv
import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as functional
import numpy as np
import torch
import random


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def gray_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("P")


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class GeneralDataset(Dataset):
    """
    A general dataset class.
    """

    def __init__(
        self,
        data_root="",
        mode="train",
        loader=default_loader,
        use_memory=True,
        trfms=None,
    ):
        """Initializing `GeneralDataset`.

        Args:
            data_root (str, optional): A CSV file with (file_name, label) records. Defaults to "".
            mode (str, optional): model mode in train/test/val. Defaults to "train".
            loader (fn, optional): specific which loader to use(see line 10-40 in this file). Defaults to default_loader.
            use_memory (bool, optional): option to use memory cache to accelerate reading. Defaults to True.
            trfms (list, optional): A transform list (in LFS, its useless). Defaults to None.
        """
        super(GeneralDataset, self).__init__()
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must be in ['train', 'val', 'test']"

        self.data_root = data_root
        self.mode = mode
        self.loader = loader
        self.use_memory = use_memory
        self.trfms = trfms

        if use_memory:
            cache_path = os.path.join(data_root, "{}.pth".format(mode))
            (
                self.data_list,
                self.label_list,
                self.class_label_dict,
            ) = self._load_cache(cache_path)
        else:
            (
                self.data_list,
                self.label_list,
                self.class_label_dict,
            ) = self._generate_data_list()

        self.label_num = len(self.class_label_dict)
        self.length = len(self.data_list)

        print(
            "load {} {} image with {} label.".format(self.length, mode, self.label_num)
        )

    def _generate_data_list(self):
        """Parse a CSV file to a data list(image_name), a label list(corresponding to the data list) and a class-label dict.

        Returns:
            tuple: A tuple of (data list, label list, class-label dict)
        """
        meta_csv = os.path.join(self.data_root, "{}.csv".format(self.mode))

        data_list = []
        label_list = []
        class_label_dict = dict()
        with open(meta_csv) as f_csv:
            f_train = csv.reader(f_csv, delimiter=",")
            for row in f_train:
                if f_train.line_num == 1:
                    continue
                image_name, image_class = row
                if image_class not in class_label_dict:
                    class_label_dict[image_class] = len(class_label_dict)
                image_label = class_label_dict[image_class]
                data_list.append(image_name)
                label_list.append(image_label)

        return data_list, label_list, class_label_dict

    def _load_cache(self, cache_path):
        """Load a pickle cache from saved file.(when use_memory option is True)

        Args:
            cache_path (str): The path to the pickle file.

        Returns:
            tuple: A tuple of (data list, label list, class-label dict)
        """
        if os.path.exists(cache_path):
            print("load cache from {}...".format(cache_path))
            with open(cache_path, "rb") as fin:
                data_list, label_list, class_label_dict = pickle.load(fin)
        else:
            print("dump the cache to {}, please wait...".format(cache_path))
            data_list, label_list, class_label_dict = self._save_cache(cache_path)

        return data_list, label_list, class_label_dict

    def _save_cache(self, cache_path):
        """Save a pickle cache to the disk.

        Args:
            cache_path (str): The path to the pickle file.

        Returns:
            tuple: A tuple of (data list, label list, class-label dict)
        """
        data_list, label_list, class_label_dict = self._generate_data_list()
        data_list = [
            self.loader(os.path.join(self.data_root, "images", path))
            for path in data_list
        ]

        with open(cache_path, "wb") as fout:
            pickle.dump((data_list, label_list, class_label_dict), fout)
        return data_list, label_list, class_label_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return a PyTorch like dataset item of (data, label) tuple.

        Args:
            idx (int): The __getitem__ id.

        Returns:
            tuple: A tuple of (image, label)
        """
        if self.use_memory:
            data = self.data_list[idx]
        else:
            image_name = self.data_list[idx]
            image_path = os.path.join(self.data_root, "images", image_name)
            data = self.loader(image_path)

        if self.trfms is not None:
            data = self.trfms(data)
        label = self.label_list[idx]

        return data, label

def crop_func(img, crop, ratio = 1.2):
    """
    Given cropping positios, relax for a certain ratio, and return new crops
    , along with the area ratio.
    """
    assert len(crop) == 4
    w,h = functional.get_image_size(img)
    if crop[0] == -1.:
        crop[0],crop[1],crop[2],crop[3]  = 0., 0., h, w
    else:
        crop[0] = max(0, crop[0]-crop[2]*(ratio-1)/2)
        crop[1] = max(0, crop[1]-crop[3]*(ratio-1)/2)
        crop[2] = min(ratio*crop[2], h-crop[0])
        crop[3] = min(ratio*crop[3], w-crop[1])
    return crop, crop[2]*crop[3]/(w*h)

class COSOCDataset(GeneralDataset):
    def __init__(self, data_root="", mode="train", loader=default_loader, use_memory=True, trfms=None, feature_image_and_crop_id='', position_list='', ratio = 1.2, crop_size = 0.08, image_sz = 84):
        super().__init__(data_root, mode, loader, use_memory, trfms)
        self.image_sz = image_sz
        self.ratio = ratio
        self.crop_size = crop_size
        with open(feature_image_and_crop_id, 'rb') as f:
            self.feature_image_and_crop_id = pickle.load(f)
        self.position_list = np.load(position_list)
        self._get_id_position_map()

    def _get_id_position_map(self):
        self.position_map = {}
        for i, feature_image_and_crop_ids in self.feature_image_and_crop_id.items():
            for clusters in feature_image_and_crop_ids:
                for image in clusters:
                    # print(image)
                    if image[0] in self.position_map:
                        self.position_map[image[0]].append((image[1],image[2]))
                    else:
                        self.position_map[image[0]] = [(image[1],image[2])]

    def _multi_crop_get(self, idx):
        if self.use_memory:
            data = self.data_list[idx]
        else:
            image_name = self.data_list[idx]
            image_path = os.path.join(self.data_root, "images", image_name)
            data = self.loader(image_path)
            ... # image -> aug(collate) -> tensor (b, patch, ...) -> classifier

        if self.trfms is not None:
            data = self.trfms(data)
        label = self.label_list[idx]

        return data, label

    def _prob_crop_get(self, idx):
        if self.use_memory:
            data = self.data_list[idx]
        else:
            image_name = self.data_list[idx]
            image_path = os.path.join(self.data_root, "images", image_name)
            data = self.loader(image_path)
            idx = int(idx)

            x = random.random()
            ran_crop_prob = 1 - torch.tensor(self.position_map[idx][0][1]).sum()
            if x > ran_crop_prob:
                crop_ids = self.position_map[idx][0][0]
                if ran_crop_prob <= x < ran_crop_prob+self.position_map[idx][0][1][0]:
                    crop_id = crop_ids[0]
                elif ran_crop_prob+self.position_map[idx][0][1][0] <= x < ran_crop_prob+self.position_map[idx][0][1][1]+self.position_map[idx][0][1][0]:
                    crop_id = crop_ids[1]
                else:
                    crop_id = crop_ids[2]
                crop = self.position_list[idx][crop_id]
                crop, space_ratio = crop_func(data, crop, ratio = self.ratio)
                data = functional.crop(data,crop[0],crop[1], crop[2],crop[3])
                data = transforms.RandomResizedCrop(self.image_sz, scale = (self.crop_size/space_ratio, 1.0))(data)
            else:
                data = transforms.RandomResizedCrop(self.image_sz)(data)

        if self.trfms is not None:
            data = self.trfms(data)
        label = self.label_list[idx]
        return data, label

    def __getitem__(self, idx):
        """Return a PyTorch like dataset item of (data, label) tuple.

        Args:
            idx (int): The __getitem__ id.

        Returns:
            tuple: A tuple of (image, label)
        """
        if self.mode == 'train':
            return self._prob_crop_get(idx)
        else:
            return self._multi_crop_get(idx)
