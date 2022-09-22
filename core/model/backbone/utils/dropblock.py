# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (
                    batch_size,
                    channels,
                    height - (self.block_size - 1),
                    width - (self.block_size - 1),
                )
            )
            if torch.cuda.is_available():
                mask = mask.cuda()
            block_mask = self._compute_block_mask(mask)
            countM = (
                block_mask.size()[0]
                * block_mask.size()[1]
                * block_mask.size()[2]
                * block_mask.size()[3]
            )
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size)
                .view(-1, 1)
                .expand(self.block_size, self.block_size)
                .reshape(-1),  # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t()
        offsets = torch.cat(
            (torch.zeros(self.block_size**2, 2).long(), offsets.long()), 1
        )
        if torch.cuda.is_available():
            offsets = offsets.cuda()

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size**2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding),
            )
            padded_mask[
                block_idxs[:, 0],
                block_idxs[:, 1],
                block_idxs[:, 2],
                block_idxs[:, 3],
            ] = 1.0
        else:
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding),
            )

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask
