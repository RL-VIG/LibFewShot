# -*- coding: utf-8 -*-
import logging
import os
from logging import config
from core.utils.utils import get_local_time

try:
    USE_RICH_CONSOLE = True
    import rich
except ImportError:
    USE_RICH_CONSOLE = False

str_level_dict = {
    "notest": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def init_logger_config(
    log_level, result_root, classifier, backbone, is_train=True, rank=0
):
    if log_level not in str_level_dict:
        raise KeyError

    level = str_level_dict[log_level]
    file_name = "{}-{}-{}-{}.log".format(
        classifier, backbone, "train" if is_train else "test", get_local_time()
    )
    log_path = os.path.join(result_root, file_name)

    if rank == 0:
        logging_config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "simple": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "level": level,
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "simple",
                    "filename": log_path,
                    "maxBytes": 100 * 1024 * 1024,
                    "backupCount": 3,
                },
            },
            "loggers": {
                "": {
                    "handlers": [
                        ("rich-console" if USE_RICH_CONSOLE else "console"),
                        "file",
                    ],
                    "level": level,
                    "propagate": True,
                }
            },
        }
    else:
        logging_config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "simple": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "": {
                    "handlers": [
                        ("rich-console" if USE_RICH_CONSOLE else "console"),
                    ],
                    "level": level,
                    "propagate": True,
                }
            },
        }

    if USE_RICH_CONSOLE:
        logging_config["handlers"].update(
            {
                "rich-console": {
                    "level": level,
                    "class": "rich.logging.RichHandler",
                }
            }
        )

    logging.config.dictConfig(logging_config)
