import os

from core.config import Config
from core.test import Test

PATH = './results/RelationNet-miniImageNet-resnet18-5-5'
VAR_DICT = {
    'test_epoch': 5,
    'device_ids': '4',
    'n_gpu': 1,
    'test_episode': 600,
    'episode_size': 1
}

if __name__ == '__main__':
    config = Config(os.path.join(PATH, 'config.yaml'), VAR_DICT).get_config_dict()
    test = Test(config, PATH)
    test.test_loop()
