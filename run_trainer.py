# -*- coding: utf-8 -*-
"""Train a LibFewShot model from a YAML config."""

import argparse
import os
import sys

sys.dont_write_bytecode = True

import torch

from core import Trainer
from core.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LibFewShot model.")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to a LibFewShot YAML config.",
    )
    args, config_args = parser.parse_known_args()
    # Config owns the framework-wide command-line options that remain.
    sys.argv = [sys.argv[0]] + config_args
    return args


def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    cli_args = parse_args()
    train_config = Config(cli_args.config).get_config_dict()

    if train_config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = train_config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=train_config["n_gpu"], args=(train_config,))
    else:
        main(0, train_config)
