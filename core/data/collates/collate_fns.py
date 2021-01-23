import itertools
from collections import Iterable

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

            images = torch.cat(images)

            targets = torch.tensor(targets,dtype=torch.int64)

            return images, targets
        except TypeError:
            raise TypeError('不应该在dataset传入transform，在collate_fn传入transform')

    def __call__(self, batch):
        return self.method(batch)


class FewShotAugCollateFn(object):
    """
    增广5次的样例: 01234 -> 0000011111222223333344444
    """

    def __init__(self, trfms, times, way_num,shot_num,query_num,episode_size):
        super(FewShotAugCollateFn, self).__init__()
        self.trfms = trfms
        self.times = 1 if times == 0 else times  # 暂时兼容times=0的写法
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.episode_size = episode_size

    def method(self, batch):
        try:
            # images
            images, labels = zip(*batch)
            # images = [img_label_tuple[0] for img_label_tuple in batch]  # 111111222222 (5s1q for example)
            images_split_by_label = [images[index:index + self.shot_num + self.query_num] for index in
                                     range(0, len(images), self.shot_num + self.query_num)]
            # 111111; 222222 ;
            images_split_by_label_type = [[spt_qry[:self.shot_num], spt_qry[self.shot_num:]] for spt_qry in
                                          images_split_by_label]
            # 11111,1;22222,2;  == [shot, query]

            # aug support # fixme: should have a elegant method # 1111111111,1;2222222222,2 # (aug_time = 2 for example)
            for cls in images_split_by_label_type:
                cls[0] = cls[0] * self.times


            # flatten and apply trfms
            flat = lambda t: [x for sub in t for x in flat(sub)] if isinstance(t, Iterable) else [t]
            images = flat(images_split_by_label_type)  # 1111111111122222222222
            images = [self.trfms(image) for image in images]  # list of tensors([c, h, w])
            images = torch.stack(images)  # [b', c, h, w] <- b' = b after aug


            # labels
            global_labels = sorted(set(labels), key=labels.index)
            global_labels = torch.tensor(global_labels,dtype=torch.int64).reshape(1,-1,1).repeat(self.episode_size,1,self.shot_num*self.times+self.query_num)


            return images, global_labels # images.shape = [e*(q+s) x c x h x w],  global_labels.shape = [e x w x (q+s)]
        except TypeError:
            raise TypeError('不应该在dataset传入transform，在collate_fn传入transform')

    def __call__(self, batch):
        return self.method(batch)
