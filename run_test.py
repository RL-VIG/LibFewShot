import os

from core.config import Config
from core.test import Test

PATH = './results/Baseline-Conv64F-5-5'
VAR_DICT = {
    'device_ids': 0,
    'test_epoch': 5
}

if __name__ == '__main__':
    config = Config(os.path.join(PATH, 'config.yaml'), VAR_DICT).get_config_dict()
    test = Test(config)
    test.test_loop()
