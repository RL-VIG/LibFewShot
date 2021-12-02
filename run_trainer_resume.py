# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer

PATH = "./results/Baseline-miniImageNet--ravi-Conv64F-5-5-Dec-01-2021-06-52-41"


def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), is_resume=True).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
