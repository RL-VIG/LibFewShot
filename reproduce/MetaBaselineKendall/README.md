## DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall's Rank Correlation

## Introduction
| Name:   | [MetaBaseline + Diffkendall]( https://arxiv.org/abs/2307.15317) |
| ------- | ------------------------------------------------------------ |
| Embed.: | ResNet12                                                     |
| Type:   | Metric                                                       |
| Venue:  | NeurIPS'23                                                   |
| Codes:  | [*Diffkendall*](https://github.com/kaipengm2/DiffKendall)    |

Cite this work with:
```bibtex
@inproceedings{NEURIPS2023_9b013332,
 author = {Zheng, Kaipeng and Zhang, Huishuai and Huang, Weiran},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {49403--49415},
 publisher = {Curran Associates, Inc.},
 title = {DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall\textquotesingle s Rank Correlation},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/9b01333262789ea3a65a5fab4c22feae-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```
---
## Results and Models

**Paper**

|   | Embedding | 💻*mini*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :computer:*mini*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) |
|---|-----------|--------------------|--------------------|--------------------|--------------------|
| 1 | ResNet12 | 65.56 ± 0.43 | 70.76 ± 0.43 | 80.79 ± 0.31 | 85.31±0.34 |

**Ours**

|   | Embedding | 💻 *mini*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :computer:*mini*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) |
|---|-----------|--------------------|--------------------|--------------------|--------------------|
| 1 | ResNet12 | 64.30 ± 0.37[:arrow_down:](https://drive.google.com/drive/folders/1hP8nFCcIHJN_2jil6BHczDXvCxMtPE1F?usp=sharing) [:clipboard:](./MetaBaselineKendall-miniImageNet--ravi-resnet12-5-1.yaml) | 69.47 ± 0.43 [:arrow_down:](https://drive.google.com/drive/folders/17b89D5GhQzNOLdqbiBiW8P8sGM5cSvxQ?usp=sharing) [:clipboard:](./MetaBaselineKendall-tiered_imagenet-resnet12-5-1.yaml) | 80.49 ± 0.25[:arrow_down:](https://drive.google.com/drive/folders/19IY3wQicsx1jvZ4trxU0oOYH3LckMGM7?usp=sharing) [:clipboard:](./MetaBaselineKendall-miniImageNet--ravi-resnet12-5-5.yaml) | 84.59 ± 0.27[:arrow_down:](https://drive.google.com/drive/folders/1oq3884J1Rce_LnnW8coTFK_82LsqgNS5) [:clipboard:](./MetaBaselineKendall-tiered_imagenet-resnet12-5-5.yaml) |

**Pretrain model**

| 💻*mini*ImageNet | :computer: *tiered*ImageNet |
| --------------- | --------------------------- |
|[:arrow_down:](https://drive.google.com/drive/folders/1UmNqCq12TFPJxnndu_pSJyHLMP5a78mC?usp=sharing) [:clipboard:](./MetabaselineKendallPretrain-miniImageNet--ravi-resnet12.yaml)                 |[:arrow_down:](https://drive.google.com/drive/folders/1e8tM9jFGsZs3c9P54Bb_19Q2T2WZ6SNe?usp=sharing) [:clipboard:](./MetabaselineKendallPretrain-tiered_imagenet-resnet12.yaml)                             |

