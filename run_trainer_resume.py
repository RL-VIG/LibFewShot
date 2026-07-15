# -*- coding: utf-8 -*-
"""Resume a LibFewShot experiment from its result directory."""

import argparse
import os
import sys

sys.dont_write_bytecode = True

import torch

from core import Trainer
from core.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Resume LibFewShot training.")
    parser.add_argument(
        "-r",
        "--result-path",
        "--result_path",
        dest="result_path",
        required=True,
        help="Training result directory containing config.yaml and checkpoints/.",
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
    config_path = os.path.join(cli_args.result_path, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError("Config file not found: {}".format(config_path))

    resume_config = Config(config_path, is_resume=True).get_config_dict()
    if resume_config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = resume_config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=resume_config["n_gpu"], args=(resume_config,))
    else:
        main(0, resume_config)
