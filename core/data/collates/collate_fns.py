import itertools

import torch


class GeneralCollateFn(object):
    def __init__(self, trfms, times):
        super(GeneralCollateFn, self).__init__()
        self.trfms = trfms
        self.times = times  # 如果不做增广，设为1

    def method(self, batch):
        try:
            images, targets = zip(*batch)

            images = list(itertools.chain.from_iterable(
                [[image] * self.times for image in images]))
            images = [self.trfms(image).unsqueeze(0) for image in images]

            targets = list(itertools.chain.from_iterable(
                [[target] * self.times for target in targets]))
            targets = [torch.tensor([target]) for target in targets]

            assert len(images) == len(targets), '图像和标签数量不一致'

            return images, targets
        except TypeError:
            raise TypeError('不应该在dataset传入transform，在collate_fn传入transform')

    def __call__(self, batch):
        return self.method(batch)


class FewShotAugCollateFn(object):
    """
    增广5次的样例: 01234 -> 0000011111222223333344444
    """

    def __init__(self, trfms, times):
        super(FewShotAugCollateFn, self).__init__()
        self.trfms = trfms
        self.times = times

    def method(self, batch):
        try:
            query_images, query_targets, support_images, support_targets = batch[0]
            query_images = [self.trfms(image).unsqueeze(0) for image in query_images]
            query_targets = [torch.tensor([target]) for target in query_targets]
            # Do aug
            aug_support_images = list(itertools.chain.from_iterable(
                [[image] * self.times for image in support_images]))
            aug_support_images = [self.trfms(image).unsqueeze(0)
                                  for image in aug_support_images]
            aug_support_targets = list(itertools.chain.from_iterable(
                [[target] * self.times for target in support_targets]))
            aug_support_targets = [torch.tensor([target])
                                   for target in aug_support_targets]

            assert len(aug_support_images) == len(aug_support_targets), \
                '图像和标签数量不一致'

            return query_images, query_targets, aug_support_images, aug_support_targets
        except TypeError:
            raise TypeError('不应该在dataset传入transform，在collate_fn传入transform')

    def __call__(self, batch):
        return self.method(batch)
