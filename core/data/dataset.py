# -*- coding: utf-8 -*-
import csv
import os
import pickle

from PIL import Image
from torch.utils.data import Dataset

try:
    import torch_dct as dct

    FGFL_DCT_AVAILABLE = True
except ImportError:
    FGFL_DCT_AVAILABLE = False
    print("torch_dct not available. FGFL frequency domain processing disabled.")


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


class FGFLDataset(GeneralDataset):
    """
    FGFL Dataset with frequency domain processing capability.
    Extends GeneralDataset to support DCT-based frequency analysis.
    """

    def __init__(
        self,
        data_root="",
        mode="train",
        loader=default_loader,
        use_memory=True,
        trfms=None,
        freq_trfms=None,
        enable_freq_domain=True,
        freq_config=None,
        dataset_name="MiniImageNet",
    ):
        """Initialize FGFLDataset.

        Args:
            freq_trfms: Additional transforms applied after frequency domain
            enable_freq_domain (bool): Whether to enable frequency processing
            freq_config (dict): Frequency processing configuration
            dataset_name (str): Name of the dataset ("MiniImageNet", "CUB", "TieredImageNet")
        """
        self.dataset_name = dataset_name
        super().__init__(data_root, mode, loader, use_memory, trfms)
        self.freq_trfms = freq_trfms
        self.enable_freq_domain = enable_freq_domain and FGFL_DCT_AVAILABLE
        self.freq_config = freq_config or {}

        if self.enable_freq_domain and not FGFL_DCT_AVAILABLE:
            print("Warning: torch_dct not available, disabling frequency")
            self.enable_freq_domain = False

    def _get_dataset_config(self):
        """Get dataset-specific configuration."""
        dataset_configs = {
            "MiniImageNet": {
                "image_dir": "images",
                "csv_format": "standard",  # filename, class
                "mean": [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                "std": [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
            },
            "CUB": {
                "image_dir": "images", 
                "csv_format": "standard",  # filename, class
                "mean": [0.485, 0.456, 0.406],  # ImageNet pretrained mean
                "std": [0.229, 0.224, 0.225],   # ImageNet pretrained std
            },
            "TieredImageNet": {
                "image_dir": "images",
                "csv_format": "standard",  # filename, class
                "mean": [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                "std": [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0],
            }
        }
        return dataset_configs.get(self.dataset_name, dataset_configs["MiniImageNet"])

    def _generate_data_list(self):
        """Parse dataset files based on dataset type."""
        dataset_config = self._get_dataset_config()
        
        if dataset_config["csv_format"] == "standard":
            return self._generate_csv_data_list(dataset_config)
        else:
            # For future extension to other formats
            return super()._generate_data_list()

    def _generate_csv_data_list(self, dataset_config):
        """Parse a CSV file for the specific dataset format."""
        meta_csv = os.path.join(self.data_root, "{}.csv".format(self.mode))
        
        data_list = []
        label_list = []
        class_label_dict = dict()
        
        with open(meta_csv) as f_csv:
            f_train = csv.reader(f_csv, delimiter=",")
            for row in f_train:
                if f_train.line_num == 1:
                    continue
                
                # Handle different CSV formats
                if len(row) >= 2:
                    image_name, image_class = row[0], row[1]
                else:
                    continue
                    
                if image_class not in class_label_dict:
                    class_label_dict[image_class] = len(class_label_dict)
                image_label = class_label_dict[image_class]
                data_list.append(image_name)
                label_list.append(image_label)

        return data_list, label_list, class_label_dict

    def _save_cache(self, cache_path):
        """Save a pickle cache to the disk with dataset-specific image path."""
        data_list, label_list, class_label_dict = self._generate_data_list()
        dataset_config = self._get_dataset_config()
        
        # Load images with dataset-specific path
        data_list = [
            self.loader(os.path.join(self.data_root, dataset_config["image_dir"], path))
            for path in data_list
        ]

        with open(cache_path, "wb") as fout:
            pickle.dump((data_list, label_list, class_label_dict), fout)
        return data_list, label_list, class_label_dict

    def _apply_frequency_filter(self, cc):
        """Apply frequency filtering based on configuration."""
        freq_type = self.freq_config.get("type", "all")
        low_cutoff = self.freq_config.get("low_freq_cutoff", 8)
        mid_cutoff = self.freq_config.get("mid_freq_cutoff", 42)

        if freq_type == "low":
            # Only low frequency
            cc[:, low_cutoff:, low_cutoff:] = 0
        elif freq_type == "mid":
            # Only mid frequency
            cc[:, :low_cutoff, :low_cutoff] = 0
            cc[:, mid_cutoff:, mid_cutoff:] = 0
        elif freq_type == "high":
            # Only high frequency
            cc[:, :mid_cutoff, :mid_cutoff] = 0
        elif freq_type == "wo_low":
            # Without low frequency
            cc[:, :low_cutoff, :low_cutoff] = 0
        elif freq_type == "wo_mid":
            # Without mid frequency
            cc[:, low_cutoff:mid_cutoff, low_cutoff:mid_cutoff] = 0
        elif freq_type == "wo_high":
            # Without high frequency
            cc[:, mid_cutoff:, mid_cutoff:] = 0
        # "all" means no filtering

        return cc

    def __getitem__(self, idx):
        """Return a data item with optional frequency domain processing."""
        dataset_config = self._get_dataset_config()
        
        if self.use_memory:
            data = self.data_list[idx]
        else:
            image_name = self.data_list[idx]
            image_path = os.path.join(self.data_root, dataset_config["image_dir"], image_name)
            data = self.loader(image_path)

        # Do NOT apply transforms here - they will be handled by collate_fn
        # Only apply frequency domain processing if enabled
        if self.enable_freq_domain:
            # Convert PIL image to tensor for DCT processing
            import torchvision.transforms as T

            to_tensor = T.ToTensor()
            data_tensor = to_tensor(data)

            # Convert to frequency domain
            cc = dct.dct_2d(data_tensor, norm="ortho")

            # Apply frequency filtering
            cc = self._apply_frequency_filter(cc)

            # Convert back to spatial domain
            data_tensor = dct.idct_2d(cc, norm="ortho")

            # Convert back to PIL for transforms in collate_fn
            to_pil = T.ToPILImage()
            data = to_pil(data_tensor)

        label = self.label_list[idx]
        return data, label
