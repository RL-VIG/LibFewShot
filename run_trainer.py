# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import torch
from core.config import Config
from core import Trainer


def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)

if __name__ == "__main__":
    config = Config("./config/proto.yaml").get_config_dict()
    
    if config["n_gpu"] > 1:
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config, ))
    else:
        main(0, config)
