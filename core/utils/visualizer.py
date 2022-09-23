# -*- coding: utf-8 -*-
from torch.utils import tensorboard


class TensorboardWriter(object):
    """
    A TensorboardWriter to write logs.
    """

    def __init__(self, log_dir):
        self.step = 0

        self.writer = tensorboard.SummaryWriter(log_dir)

        self.tb_writer_funcs = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }

    def set_step(self, step):
        self.step = step

    def __getattr__(self, name):
        """

        :param name:
        :return:
        """
        if name in self.tb_writer_funcs:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            raise RuntimeError

    def close(
        self,
    ):
        self.writer.close()
