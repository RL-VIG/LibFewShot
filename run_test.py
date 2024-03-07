# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "./results/FRN--resnet12-15-5-Dec-01-2023-19-59-46"
VAR_DICT = {
    # "seed": 42,
    "test_epoch": 2,
    "device_ids": "6",
    # "n_gpu": 1,
    "test_episode": 400,
    "test_shot": 1,
    "episode_size": 1,
    # "data_root": "/data/fewshot/tiered_imagenet"
}


def main(rank, config):
    test = Test(rank, config, PATH)
    test.test_loop()


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
