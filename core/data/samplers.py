import numpy as np
import torch
from torch.utils.data import Sampler


class CategoriesSampler(Sampler):

    def __init__(self, label, episode_size, episode_num, way_num, image_num):
        super(CategoriesSampler, self).__init__(label)

        self.episode_size = episode_size
        self.episode_num = episode_num
        self.way_num = way_num
        self.image_num = image_num

        label = np.array(label)
        self.idx_list = []
        for label_idx in range(max(label) + 1):
            ind = np.argwhere(label == label_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)

    def __len__(self):
        return self.episode_num

    def __iter__(self):
        batch = []
        for i_batch in range(self.episode_num):
            classes = torch.randperm(len(self.idx_list))[:self.way_num]
            for c in classes:
                idxes = self.idx_list[c.item()]
                pos = torch.randperm(idxes.size(0))[:self.image_num]
                batch.append(idxes[pos])
            if len(batch) == self.episode_size * self.way_num:
                batch = torch.stack(batch).reshape(-1)
                yield batch
                batch = []
