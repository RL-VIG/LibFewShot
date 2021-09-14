# -*- coding: utf-8 -*-
# Adapted from https://github.com/uoguelph-mlrg/Cutout.
import torch
import numpy as np

#         >>> transform=transforms.Compose([
#         >>>     transforms.Resize(256),
#         >>>     Cutout(n_holes=args.n_holes, length=args.length)
#         >>>     transforms.ToTensor()])


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=1):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = np.asarray(img)

        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = np.expand_dims(mask, 2).repeat(3, axis=2)

        img = img * mask

        return img
