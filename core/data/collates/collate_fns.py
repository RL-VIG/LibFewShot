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
            return query_images, query_targets, support_images, support_targets
        except TypeError:
            raise TypeError('不应该在dataset传入transform，在collate_fn传入transform')

    def __call__(self, batch):
        # batch:list, batch[0]:tuple(query_images:[]PIL, query_targets[]int64, support_images:[]PIL, support_targets[
        # ]int64)
        return self.method(batch)
