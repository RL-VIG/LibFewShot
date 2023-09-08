# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os
import torch
from core.config import Config
from core import Test


PATH = "./resultsDF20/ProtoNet-DF20-Conv64F-5-5-May-03-2023-20-32-25"
VAR_DICT = {
    "test_epoch": 1,
    "device_ids": "0",
    "n_gpu": 1,
    "test_episode": 1,
    "episode_size": 1,
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
