# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "./results/MetaBaseline-miniImageNet--ravi-resnet12-5-1-Apr-30-2024-13-31-39"
# VAR_DICT = {
#     "test_epoch": 5,
#     "device_ids": "4,5",
#     "n_gpu": 2,
#     "test_episode": 600,
#     "episode_size": 2,
# }


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml")).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
