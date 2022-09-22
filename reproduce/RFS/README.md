# Rethinking Few-Shot Image Classification: A Good Embedding is All You Need?
## Introduction
| Name:    | [RFS](https://arxiv.org/abs/2003.11539)                          |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ResNet18 |
| Type:    | Fine-tuning       |
| Venue:   | ECCV'20                      |
| Codes:   | [**rfs**](https://github.com/WangYueFt/rfs)|

+ When testing the `RFS-simple`, you need to change `test-shot` to `1 or 5` for different setting with 1 checkpoint.
+ Notice that in `Table.2`, we do not use Test-DA(which augment the test samples with 5-times) for fair.

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/eccv/TianWKTI20,
  author    = {Yonglong Tian and
               Yue Wang and
               Dilip Krishnan and
               Joshua B. Tenenbaum and
               Phillip Isola}
  title     = {Rethinking Few-Shot Image Classification: {A} Good Embedding is All
               You Need?},
  booktitle = {Computer Vision - {ECCV} 2020 - 16th European Conference, Glasgow,
               UK, August 23-28, 2020, Proceedings, Part {XIV}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12359},
  pages     = {266--282},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-58568-6_16},
  doi       = {10.1007/978-3-030-58568-6_16}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12  [^metaopt] | 62.02 ± 0.63 | 62.80 ± 0.52 [:arrow_down:](https://drive.google.com/drive/folders/1COMUhto08xtSaOazlMw1GndEAfeIZ-XQ?usp=sharing)  [:clipboard:](./RFS-simple-miniImageNet-ResNet12M-Table1.yaml) | 79.64 ± 0.44 | 79.57± 0.39 [:arrow_down:](https://drive.google.com/drive/folders/1COMUhto08xtSaOazlMw1GndEAfeIZ-XQ?usp=sharing) [:clipboard:](./RFS-simple-miniImageNet-ResNet12M-Table1.yaml) | rfs-simple-Table-1 |
| 2 | Conv64F | - | 47.97 ± 0.33 [:arrow_down:](https://drive.google.com/drive/folders/1K-8r4DtVzFadYWWhWSuneWPcQHRbReLl?usp=sharing)  [:clipboard:](./RFS-simple-miniImageNet--ravi-Conv64F-Table2.yaml) | - | 65.88 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1K-8r4DtVzFadYWWhWSuneWPcQHRbReLl?usp=sharing) [:clipboard:](./RFS-simple-miniImageNet--ravi-Conv64F-Table2.yaml) | Table.2 |
| 3 | ResNet12 | - |  61.65 ± 0.35 [:arrow_down:](https://drive.google.com/drive/folders/1BstYDeMzK4N2jhU3uXTM4t02M1LrKyUN?usp=sharing)  [:clipboard:](./RFS-simple-miniImageNet--ravi-resnet12-Table2.yaml) | - | 78.88 ± 0.25 [:arrow_down:](https://drive.google.com/drive/folders/1BstYDeMzK4N2jhU3uXTM4t02M1LrKyUN?usp=sharing) [:clipboard:](./RFS-simple-miniImageNet--ravi-resnet12-Table2.yaml) | Table.2 |
| 4 | ResNet18 | - | 61.65 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1WR3Z2LSwN6Hzf10QO1Wn-pkfvviltNcR?usp=sharing)  [:clipboard:](./RFS-simple-miniImageNet--ravi-resnet18-Table2.yaml) | - | 76.60 ± 0.28 [:arrow_down:](https://drive.google.com/drive/folders/1WR3Z2LSwN6Hzf10QO1Wn-pkfvviltNcR?usp=sharing) [:clipboard:](./RFS-simple-miniImageNet--ravi-resnet18-Table2.yaml) | Table.2 |

|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 52.21 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/14rJGCWpjrcWINvGAO6bPv1gV70MsLb4S?usp=sharing)  [:clipboard:](./RFS-simple-tiered_imagenet-Conv64F-Table2.yaml) | - | 71.82 ± 0.32 [:arrow_down:](https://drive.google.com/drive/folders/14rJGCWpjrcWINvGAO6bPv1gV70MsLb4S?usp=sharing) [:clipboard:](./RFS-simple-tiered_imagenet-Conv64F-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 70.55 ± 0.42 [:arrow_down:](https://drive.google.com/drive/folders/1vht82-xCd191LH95uhPKDXEyv6TbLNxT?usp=sharing)  [:clipboard:](./RFS-simple-tiered_imagenet-resnet12-Table2.yaml) | - | 84.74 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1vht82-xCd191LH95uhPKDXEyv6TbLNxT?usp=sharing) [:clipboard:](./RFS-simple-tiered_imagenet-resnet12-Table2.yaml) | Table.2 |
| 3 | ResNet18| - | 69.14 ± 0.42 [:arrow_down:](https://drive.google.com/drive/folders/1w4A1ttxZ7oq9GuMq-1nNacufqWO0uPvl?usp=sharing)  [:clipboard:](./RFS-simple-tiered_imagenet-resnet18-Table2.yaml) | - | 83.21 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1w4A1ttxZ7oq9GuMq-1nNacufqWO0uPvl?usp=sharing) [:clipboard:](./RFS-simple-tiered_imagenet-resnet18-Table2.yaml) | Table.2 |

[^metaopt]: ResNet12-MetaOpt with [64,160,320,640].
