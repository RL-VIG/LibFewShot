from .autoaugment import ImageNetPolicy
from .cutout import Cutout
from .randaugment import RandAugment
from torchvision import transforms
CJ_DICT = {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4}

def get_augment_method(config,):
    if 'augment_method' not in config or config['augment_method'] == 'NormalAug':
        trfms = [transforms.ColorJitter(**CJ_DICT),
                 transforms.RandomHorizontalFlip()]
    elif config['augment_method'] == 'AutoAugment':
        trfms = [ImageNetPolicy()]
    elif config['augment_method'] == 'Cutout':
        trfms = [Cutout()]
    elif config['augment_method'] == 'RandAugment':
        trfms = [RandAugment()]

    return trfms