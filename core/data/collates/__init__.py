from .collate_functions import *
from .contrib import get_augment_method
from ...utils import ModelType


def get_collate_function(config, trfms, mode, model_type):
    """
    通过配置文件设置相应的`collate_fn`

    对于pretrain方法的train阶段dataloader，返回通用collate_fn（即正常情况下的collate_fn），对于其他方法以及pretrain的test,val阶段dataloader，按照对应的way/shoy/query设置返回对应的collate_fn。

    Args:
        config (dict): A LFS setting dict.
        trfms (list): A torchvision transform list.
        mode (str): model mode in ['train', 'test', 'val']
        model_type (ModelType): An ModelType enum value of model.

    Returns:
        [type]: [description]
    """
    assert model_type != ModelType.ABSTRACT
    if mode == "train" and model_type == ModelType.PRETRAIN:
        collate_function = GeneralCollateFunction(
            trfms, config["augment_times"])
    else:
        collate_function = FewShotAugCollateFunction(
            trfms,
            config["augment_times"],
            config["way_num"] if mode == "train" else config["test_way"],
            config["shot_num"] if mode == "train" else config["test_shot"],
            config["query_num"] if mode == "train" else config["test_query"],
            config["episode_size"],
        )

    return collate_function
