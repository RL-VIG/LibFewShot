# -*- coding: utf-8 -*-
import csv
import os
import pickle

from PIL import Image
from torch.utils.data import Dataset


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
