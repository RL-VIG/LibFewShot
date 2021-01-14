import os

from core.config import Config
from core.test import Test

PATH = './results/DN4-Conv64FLeakyReLU-5-5'
VAR_DICT = {
    'test_epoch': 5,
    'device_ids': '2,3',
    'n_gpu': 2,
    'test_episode': 600,
    'episode_size': 4
}

if __name__ == '__main__':
    config = Config(os.path.join(PATH, 'config.yaml'), VAR_DICT).get_config_dict()
    test = Test(config)
    test.test_loop()
