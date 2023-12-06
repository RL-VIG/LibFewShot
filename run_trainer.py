# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer


def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    config = Config("./config/metal.yaml").get_config_dict()
    ascii_art = '''
    #                       _oo0oo_
    #                      o8888888o
    #                      88" . "88
    #                      (| -_- |)
    #                      0\\  =  /0
    #                    ___/`---'\\___
    #                  .' \\\\|     |// '.
    #                 / \\\\|||  :  |||// \\
    #                / _||||| -:- |||||- \\
    #               |   | \\\\\\  -  /// |   |
    #               | \\_|  ''\\---/''  |_/ |
    #               \\  .-\\__  '-'  ___/-. /
    #             ___'. .'  /--.--\\  `. .'___
    #          ."" '<  `.___\\_<|>_/___.' >' "".
    #         | | :  `- \\`.;`\\ _ /`;.`/ - ` : | |
    #         \\  \\ `_.   \\_ __\\ /__ _/   .-` /  /
    #     =====`-.____`.___ \\_____/___.-`___.-'=====
    #                       `=---='
    #
    #
    #     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    #               佛祖保佑         永无BUG
    '''
    print(ascii_art)
    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)
