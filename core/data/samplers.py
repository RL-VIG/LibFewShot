# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


def get_sampler(dataset, few_shot, distribute, mode, config):
    if few_shot:
        # Check if FGFL mode is enabled by classifier name
        classifier_name = config.get("classifier", {}).get("name", "")
        use_fgfl = "GAIN" in classifier_name.upper()

        print(f"Sampler debug: classifier_name={classifier_name}, use_fgfl={use_fgfl}")

        if distribute:
            sampler = DistributedCategoriesSampler(
                label_list=dataset.label_list,
                label_num=dataset.label_num,
                episode_size=config["episode_size"] // config["n_gpu"],
                episode_num=(
                    (
                        config["train_episode"]
                        if mode == "train"
                        else config["test_episode"]
                    )
                    // config["n_gpu"]
                ),
                way_num=config["way_num"] if mode == "train" else config["test_way"],
                image_num=(
                    config["shot_num"] + config["query_num"]
                    if mode == "train"
                    else config["test_shot"] + config["test_query"]
                ),
                rank=config["rank"],
                seed=0,
                world_size=config["n_gpu"],
            )
        else:
            # Choose sampler based on FGFL mode
            sampler_class = FGFLCompatibleSampler if use_fgfl else CategoriesSampler
            sampler = sampler_class(
                label_list=dataset.label_list,
                label_num=dataset.label_num,
                episode_size=config["episode_size"],
                episode_num=(
                    config["train_episode"]
                    if mode == "train"
                    else config["test_episode"]
                ),
                way_num=config["way_num"] if mode == "train" else config["test_way"],
                image_num=(
                    config["shot_num"] + config["query_num"]
                    if mode == "train"
                    else config["test_shot"] + config["test_query"]
                ),
            )
    else:
        if distribute:
            sampler = DistributedSampler(dataset, rank=config["rank"], shuffle=True)
        else:
            sampler = None
    return sampler


class CategoriesSampler(Sampler):
    """A Sampler to sample a FSL task.

    Args:
        Sampler (torch.utils.data.Sampler): Base sampler from PyTorch.
    """

    def __init__(
        self,
        label_list,
        label_num,
        episode_size,
        episode_num,
        way_num,
        image_num,
    ):
        """Init a CategoriesSampler and generate a label-index list.

        Args:
            label_list (list): The label list from label list.
            label_num (int): The number of unique labels.
            episode_size (int): FSL setting.
            episode_num (int): FSL setting.
            way_num (int): FSL setting.
            image_num (int): FSL setting.
        """
        super(CategoriesSampler, self).__init__(label_list)

        self.episode_size = episode_size
        self.episode_num = episode_num
        self.way_num = way_num
        self.image_num = image_num

        label_list = np.array(label_list)
        self.idx_list = []
        for label_idx in range(label_num):
            ind = np.argwhere(label_list == label_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)

    def __len__(self):
        return self.episode_num // self.episode_size

    def __iter__(self):
        """Random sample a FSL task batch(multi-task).

        Yields:
            torch.Tensor: The stacked tensor of a FSL task batch(multi-task).
        """
        batch = []
        for i_batch in range(self.episode_num):
            classes = torch.randperm(len(self.idx_list))[: self.way_num]
            for c in classes:
                idxes = self.idx_list[c.item()]
                pos = torch.randperm(idxes.size(0))[: self.image_num]
                batch.append(idxes[pos])
            if len(batch) == self.episode_size * self.way_num:
                # TODO
                batch = torch.stack(batch).reshape(-1)
                yield batch
                batch = []


class FGFLCompatibleSampler(CategoriesSampler):
    """FGFL兼容的sampler，使用.t().reshape(-1)来保持原始行为

    FGFL使用shot-wise的数据组织方式，这对于其对比学习和triplet loss
    训练策略是必要的。
    """

    def __iter__(self):
        """Random sample a FSL task batch with FGFL-compatible ordering.

        FGFL ordering: shot-wise organization
        - 第1组包含所有类别的第1个样本
        - 第2组包含所有类别的第2个样本
        - 这种方式有利于类间对比学习

        Yields:
            torch.Tensor: The stacked tensor with FGFL-compatible ordering.
        """
        batch = []
        for i_batch in range(self.episode_num):
            classes = torch.randperm(len(self.idx_list))[: self.way_num]
            for c in classes:
                idxes = self.idx_list[c.item()]
                pos = torch.randperm(idxes.size(0))[: self.image_num]
                batch.append(idxes[pos])
            if len(batch) == self.episode_size * self.way_num:
                # 调试信息 - 只在第一个episode显示
                if i_batch == 0:
                    print("=== FGFL Sampler Debug ===")
                    print(f"Episode {i_batch}: Selected classes {classes}")
                    before_transpose = torch.stack(batch)
                    print(f"Before transpose shape: {before_transpose.shape}")
                    print(
                        f"Before transpose (class-wise): "
                        f"{before_transpose.reshape(-1)[:10]}"
                    )

                # 使用FGFL的reshape方式: transpose + reshape
                # 这会改变数据组织从class-wise到shot-wise
                batch = torch.stack(batch).t().reshape(-1)

                # if i_batch == 0:
                #     print(f"After transpose (shot-wise): {batch[:10]}")
                #     print("FGFL shot-wise: 前5个来自各类第1个样本")
                #     print("=========================")

                yield batch
                batch = []


class DistributedCategoriesSampler(Sampler):
    """A Sampler to sample a FSL task for DDP.

    Args:
        Sampler (torch.utils.data.Sampler): Base sampler from PyTorch.
    """

    def __init__(
        self,
        label_list,
        label_num,
        episode_size,
        episode_num,
        way_num,
        image_num,
        rank,
        seed=0,
        world_size=1,
    ):
        """Init a CategoriesSampler and generate a label-index list.

        Args:
            label_list (list): The label list from label list.
            label_num (int): The number of unique labels.
            episode_size (int): FSL setting.
            episode_num (int): FSL setting.
            way_num (int): FSL setting.
            image_num (int): FSL setting.
        """
        super(DistributedCategoriesSampler, self).__init__(label_list)

        self.episode_size = episode_size
        self.episode_num = episode_num
        self.way_num = way_num
        self.image_num = image_num
        self.rank = rank
        self.seed = seed
        self.world_size = world_size
        self.epoch = 0

        label_list = np.array(label_list)
        self.idx_list = []
        for label_idx in range(label_num):
            ind = np.argwhere(label_list == label_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)

        self.cls_g = torch.Generator()
        self.img_g = torch.Generator()

        self.cls_g.manual_seed(self.cls_g.seed())
        self.img_g.manual_seed(self.img_g.seed())

        # print(f"test torch seed {torch.randn(1)}")
        # print(f"cuda{rank} clsg init seed {self.cls_g.initial_seed()}")
        # print(f"cuda{rank} imgg init seed {self.img_g.initial_seed()}")

    def __len__(self):
        return self.episode_num // self.episode_size

    def __iter__(self):
        """Random sample a FSL task batch(multi-task).

        Yields:
            torch.Tensor: The stacked tensor of a FSL task batch(multi-task).
        """
        batch = []
        for i_batch in range(self.episode_num):
            classes = torch.randperm(len(self.idx_list), generator=self.cls_g)[
                : self.way_num
            ]
            for c in classes:
                idxes = self.idx_list[c.item()]
                pos = torch.randperm(idxes.size(0), generator=self.img_g)[
                    : self.image_num
                ]
                batch.append(idxes[pos])
            if len(batch) == self.episode_size * self.way_num:
                batch = torch.stack(batch).reshape(-1)
                yield batch
                batch = []

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        # self.cls_g.manual_seed(self.seed + self.epoch)
        # # FIXME not so random, 10000 means no method could train 10000 epochs, so cls_g will not have the same seed with img_g
        # self.img_g.manual_seed(self.seed + self.epoch + 10000)
