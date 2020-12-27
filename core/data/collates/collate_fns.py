import itertools
import torch

class VanillaCollateFn(object):
    """
    VanillaCollateFn 只做transform，不关心augment
    """

    def __init__(self, trfms):
        super(VanillaCollateFn, self).__init__()
        self.trfms = trfms

    def method(self, batch):
        try:
            query_images, query_targets, support_images, support_targets = batch[0]
            query_images = [self.trfms(image).unsqueeze(0) for image in query_images]
            # unsqueeze: n_dim=3 -> n_dim=4，如果使用原来的方法，dataloader应该返回的是n_dim=4
            support_images = [self.trfms(image).unsqueeze(0) for image in support_images]
            query_targets = [torch.tensor([target]) for target in query_targets]
            support_targets = [torch.tensor([target]) for target in support_targets]
            return query_images, query_targets, support_images, support_targets
        except TypeError:
            raise TypeError('不应该在dataset传入transform，在collate_fn传入transform')

    def __call__(self, batch):
        # batch:list,
        # batch[0]:tuple(query_images:[]PIL, query_targets[]int64, support_images:[]PIL, support_targets[]int64)
        return self.method(batch)


class SampleFiveTimesAugCollateFn(object):
    """
    增广5次的样例: 01234 -> 0000011111222223333344444
    """
    def __init__(self, trfms):
        super(SampleFiveTimesAugCollateFn, self).__init__()
        self.trfms = trfms
        self.trfms_aug = trfms
        self.times = 5

    def method(self, batch):
        try:
            query_images, query_targets, support_images, support_targets = batch[0]
            query_images = [self.trfms(image).unsqueeze(0) for image in query_images]
            # Do aug
            aug_support_images = list(itertools.chain.from_iterable([[image]*self.times for image in support_images]))
            aug_support_images = [self.trfms(image).unsqueeze(0) for image in support_images]
            aug_support_targets = list(itertools.chain.from_iterable([[target]*self.times for target in support_targets]))
            return query_images, query_targets, aug_support_images, aug_support_targets
        except TypeError:
            raise TypeError('不应该在dataset传入transform，在collate_fn传入transform')

    def __call__(self, batch):
        return self.method(batch)
