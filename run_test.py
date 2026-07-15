# -*- coding: utf-8 -*-
"""Evaluate a trained LibFewShot experiment."""

import argparse
import os
import sys

sys.dont_write_bytecode = True

import torch

from core import Test
from core.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LibFewShot result directory.")
    parser.add_argument(
        "-r",
        "--result-path",
        "--result_path",
        dest="result_path",
        required=True,
        help="Training result directory containing config.yaml and checkpoints/.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Optional config file. Defaults to RESULT_PATH/config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename inside RESULT_PATH/checkpoints/.",
    )
    parser.add_argument("--data-root", "--data_root", dest="data_root")
    parser.add_argument("--test-way", "--test_way", dest="test_way", type=int)
    parser.add_argument("--test-shot", "--test_shot", dest="test_shot", type=int)
    parser.add_argument("--test-query", "--test_query", dest="test_query", type=int)
    parser.add_argument("--test-episode", "--test_episode", dest="test_episode", type=int)
    parser.add_argument("--test-epoch", "--test_epoch", dest="test_epoch", type=int)
    parser.add_argument("--episode-size", "--episode_size", dest="episode_size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device-ids", "--device_ids", dest="device_ids")
    parser.add_argument("--n-gpu", "--n_gpu", dest="n_gpu", type=int)
    parser.add_argument("--seed", type=int)

    args, config_args = parser.parse_known_args()
    # Config owns the framework-wide command-line options that remain.
    sys.argv = [sys.argv[0]] + config_args
    return args


def build_config(args):
    config_path = args.config or os.path.join(args.result_path, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError("Config file not found: {}".format(config_path))

    overrides = {
        "data_root": args.data_root,
        "test_way": args.test_way,
        "test_shot": args.test_shot,
        "test_query": args.test_query,
        "test_episode": args.test_episode,
        "test_epoch": args.test_epoch,
        "episode_size": args.episode_size,
        "workers": args.workers,
        "device_ids": args.device_ids,
        "n_gpu": args.n_gpu,
        "seed": args.seed,
        "test_checkpoint": args.checkpoint,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return Config(config_path, overrides).get_config_dict()


def main(rank, config, result_path):
    tester = Test(rank, config, result_path)
    tester.test_loop()


if __name__ == "__main__":
    cli_args = parse_args()
    test_config = build_config(cli_args)

    if test_config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = test_config["device_ids"]
        torch.multiprocessing.spawn(
            main,
            nprocs=test_config["n_gpu"],
            args=(test_config, cli_args.result_path),
        )
    else:
        main(0, test_config, cli_args.result_path)
