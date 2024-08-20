# Class-Aware Patch Embedding Adaptation for Few-Shot Image Classification
## Introduction
| Name:    | [CPEA](https://openaccess.thecvf.com/content/ICCV2023/papers/Hao_Class-Aware_Patch_Embedding_Adaptation_for_Few-Shot_Image_Classification_ICCV_2023_paper.pdf)          |
|----------|-------------------------------|
| Embed.:  | ViT-small |
| Type:    | Metric       |
| Venue:   | ICCV'23                      |
| Codes:   | [*CPEA*](https://github.com/FushengHao/CPEA)                   |
| Backbone: | [*ViT(pretrained on specific datasets)*](https://github.com/mrkshllr/FewTURE) |

Cite this work with:
```bibtex
@InProceedings{Hao_2023_ICCV,
    author    = {Hao, Fusheng and He, Fengxiang and Liu, Liu and Wu, Fuxiang and Tao, Dacheng and Cheng, Jun},
    title     = {Class-Aware Patch Embedding Adaptation for Few-Shot Image Classification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18905-18915}
}
```
---
## Results and Models

**Paper**

|   | Embedding | 💻: *mini*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :computer:*mini*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) |
|---|-----------|--------------------|--------------------|--------------------|--------------------|
| 1 | ViT-small | 71.97 ± 0.65 | 76.93 ± 0.70 | 87.06 ± 0.38 | 90.12±0.45 |

**Ours**

|   | Embedding | 💻: *mini*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :computer:*mini*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) |
|---|-----------|--------------------|--------------------|--------------------|--------------------|
| 1 | ViT-small | 72.484 [:arrow_down:](https://drive.google.com/drive/folders/1mAHEnQ9AZbm8ILbU8hQa1V1l_h-i8Bjp?usp=sharing) [:clipboard:](https://github.com/Cbphcr/LibFewShot/blob/add-method-cpea-backbone-VitClassAware/reproduce/CPEA/CPEANet-miniImageNet--ravi-VisionTransformer-5-1.yaml) | 77.484 [:arrow_down:](https://drive.google.com/drive/folders/1d6Rm8-QwcDLIohAjkSx6vdGwNuYovAD2?usp=sharing) [:clipboard:](https://github.com/Cbphcr/LibFewShot/blob/add-method-cpea-backbone-VitClassAware/reproduce/CPEA/CPEANet-tiered_imagenet-VisionTransformer-5-1.yaml) | 87.734 [:arrow_down:](https://drive.google.com/drive/folders/1mAHEnQ9AZbm8ILbU8hQa1V1l_h-i8Bjp?usp=sharing) [:clipboard:](https://github.com/Cbphcr/LibFewShot/blob/add-method-cpea-backbone-VitClassAware/reproduce/CPEA/CPEANet-miniImageNet--ravi-VisionTransformer-5-5.yaml) | 90.139 [:arrow_down:](https://drive.google.com/drive/folders/1v3hfYSO4HjIC1JMLnOj6AjzyYLd2t7yJ?usp=sharing) [:clipboard:](https://github.com/Cbphcr/LibFewShot/blob/add-method-cpea-backbone-VitClassAware/reproduce/CPEA/CPEANet-tiered_imagenet-VisionTransformer-5-5.yaml) |
