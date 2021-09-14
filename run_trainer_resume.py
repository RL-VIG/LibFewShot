# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os

from core.config import Config
from core import Trainer

PATH = "./results/RFSModel-miniImageNet--ravi-resnet12-5-1-KD2-Aug-28-2021-14-47-07"

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), is_resume=True).get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
