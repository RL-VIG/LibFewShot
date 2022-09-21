# A Closer Look at Few-shot Classification
## Introduction
| Name:    | [Baseline++](https://arxiv.org/abs/1904.04232)                          |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ResNet18 |
| Type:    | Fine-tuning       |
| Venue:   | ICLR'19                      |
| Codes:   | [**CloserLookFewShot**](https://github.com/wyharveychen/CloserLookFewShot)|

+ When reproduceingthis method with the same setting in the original paper, you should skip validation during training-stage and choose the last model it saves.

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/iclr/ChenLKWH19,
  author    = {Wei{-}Yu Chen and
               Yen{-}Cheng Liu and
               Zsolt Kira and
               Yu{-}Chiang Frank Wang and
               Jia{-}Bin Huang},
  title     = {A Closer Look at Few-shot Classification},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=HkxLXnAcFQ}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 48.86 ± 0.35 [:arrow_down:](https://drive.google.com/drive/folders/1PTrmgQYCeInx4zdbre3a9JSgZM9abMGv?usp=sharing) [:clipboard:](./BaselinePlus-miniImageNet--ravi-Conv64F-Table2.yaml) | - | 63.29 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1PTrmgQYCeInx4zdbre3a9JSgZM9abMGv?usp=sharing) [:clipboard:](./BaselinePlus-miniImageNet--ravi-Conv64F-Table2.yaml) | Table2 |
| 2 | ResNet12 | - | 56.75 ± 0.38 [:arrow_down:](https://drive.google.com/drive/folders/1oU4qepvyfiduzXSAHsD7Bc9paOyCzaGY?usp=sharing) [:clipboard:](./BaselinePlus-miniImageNet--ravi-resnet12-Table2.yaml) | - | 66.36 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1oU4qepvyfiduzXSAHsD7Bc9paOyCzaGY?usp=sharing) [:clipboard:](./BaselinePlus-miniImageNet--ravi-resnet12-Table2.yaml) | Table2 |
| 3 | Conv64F | 48.24 | 46.21 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1K6vqkh-bm0StFyT3R5YCIn3CpUQ91zwn?usp=sharing) [:clipboard:](./BaselinePlus-miniImageNet--ravi-Conv64F-5-Reproduce.yaml) | 66.43 | 65.18 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1K6vqkh-bm0StFyT3R5YCIn3CpUQ91zwn?usp=sharing) [:clipboard:](./BaselinePlus-miniImageNet--ravi-Conv64F-5-Reproduce.yaml) | Reproduce |


|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 55.94 ± 0.39 [:arrow_down:](https://drive.google.com/drive/folders/1mV-oD12E-vW_d2VXVvBgNxvIe9Bg9-Xj?usp=sharing) [:clipboard:](./BaselinePlus-tiered_imagenet-Conv64F-Table2.yaml) | - | 73.80 ± 0.32 [:arrow_down:](https://drive.google.com/drive/folders/1mV-oD12E-vW_d2VXVvBgNxvIe9Bg9-Xj?usp=sharing) [:clipboard:](./BaselinePlus-tiered_imagenet-Conv64F-Table2.yaml) | Table2 |
| 2 | ResNet12 | - | 65.95 ± 0.42 [:arrow_down:](https://drive.google.com/drive/folders/1Cl8UAMxVU80YWLY4ZkgR0LG5L7Ys8f4y?usp=sharing) [:clipboard:](./BaselinePlus-tiered_imagenet-resnet12-Table2.yaml) | - | 82.25 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1Cl8UAMxVU80YWLY4ZkgR0LG5L7Ys8f4y?usp=sharing) [:clipboard:](./BaselinePlus-tiered_imagenet-resnet12-Table2.yaml) | Table2 |
