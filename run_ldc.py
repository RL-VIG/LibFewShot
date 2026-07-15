# -*- coding: utf-8 -*-
"""Train, validate, or test the paper-style LDC integration."""

import argparse
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core.model.metric.ldc import is_ldc_paper_config, run_ldc_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LDC paper reproduction.")
    parser.add_argument(
        "-c",
        "--config",
        default="./config/ldc.yaml",
        help="Path to the LDC YAML config.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--eval-only",
        "--eval_only",
        dest="eval_only",
        action="store_true",
        help="Evaluate a trained checkpoint instead of training.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for --eval-only (defaults to OUTPUT_DIR/model_best.pth).",
    )
    mode.add_argument(
        "--check-data",
        "--check_data",
        dest="check_data",
        action="store_true",
        help="Validate dataset paths and exit.",
    )
    mode.add_argument(
        "--smoke-forward",
        "--smoke_forward",
        dest="smoke_forward",
        action="store_true",
        help="Run one forward pass and exit.",
    )
    args, config_args = parser.parse_known_args()
    if args.checkpoint and not args.eval_only:
        parser.error("--checkpoint can only be used with --eval-only")
    sys.argv = [sys.argv[0]] + config_args
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    ldc_config = Config(cli_args.config).get_config_dict()
    if not is_ldc_paper_config(ldc_config):
        raise ValueError("The selected config must use classifier.name=LDC and paper_mode=true.")
    run_ldc_from_config(
        ldc_config,
        eval_only=cli_args.eval_only,
        checkpoint=cli_args.checkpoint,
        check_data=cli_args.check_data,
        smoke_forward=cli_args.smoke_forward,
    )
