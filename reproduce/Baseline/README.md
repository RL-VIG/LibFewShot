# A Closer Look at Few-shot Classification
## Introduction
| Name:    | [Baseline](https://arxiv.org/abs/1904.04232)                          |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ResNet18 |
| Type:    | Fine-tuning       |
| Venue:   | ICLR'19                      |
| Codes:   | [**CloserLookFewShot**](https://github.com/wyharveychen/CloserLookFewShot)|

+ When reproduceingthis method with the same setting in the original paper, you should skip validation during training-stage and choose the last model it saves.
+ Notice that baseline use N-cls-head to train, where N > num_base_classes.

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
| 1 | Conv64F | 42.11 | 42.34 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1GKt_Y-CZqgzsm4YQEkeP0j_xTvjKhlng?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-Conv64F-5-Reproduce.yaml) | 62.53 | 62.18 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1GKt_Y-CZqgzsm4YQEkeP0j_xTvjKhlng?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-Conv64F-5-Reproduce.yaml) | Reproduce |
| 2 | Conv64F | - | 44.90 ± 0.32 [:arrow_down:](https://drive.google.com/drive/folders/14KXSJnGnX7TCp3-h0IRPKDKgKDQlqy8N?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-Conv64F-Table2.yaml) | - | 63.96 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/14KXSJnGnX7TCp3-h0IRPKDKgKDQlqy8N?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-Conv64F-Table2.yaml) | Table.2 |
| 3 | ResNet12 | - | 56.39 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/1iDO-nbK2Xo5pGODJ88uzEZ5hVV6w2dKf?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-resnet12-Table2.yaml) | - | 76.18 ± 0.27 [:arrow_down:](https://drive.google.com/drive/folders/1iDO-nbK2Xo5pGODJ88uzEZ5hVV6w2dKf?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-resnet12-Table2.yaml) | Table.2 |
| 4 | ResNet18 | - | 54.11 ± 0.35 [:arrow_down:](https://drive.google.com/drive/folders/1-EYAms9N_iRXHu14JHigbIqAsMTEJXgG?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-resnet18-Table2.yaml) | - | 74.44 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1-EYAms9N_iRXHu14JHigbIqAsMTEJXgG?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-resnet18-Table2.yaml) | Table.2 |
| 5 | ResNet18 | - | 51.18 ± 0.34 [:arrow_down:](https://drive.google.com/drive/folders/1u3Xbog183WrQGm8durg6W79uvkpVQPst?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-resnet18-5-Reproduce.yaml) | - | 74.06 ± 0.28 [:arrow_down:](https://drive.google.com/drive/folders/1u3Xbog183WrQGm8durg6W79uvkpVQPst?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-resnet18-5-Reproduce.yaml) | Reproduce |




|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 48.20 ± 0.35 [:arrow_down:](https://drive.google.com/drive/folders/1WI-oqDQz73j28oMcFe2h_HxMv-oN9kH9?usp=sharing) [:clipboard:](./Baseline-tiered_imagenet-Conv64F-Table2.yaml) | - | 68.96 ± 0.33 [:arrow_down:](https://drive.google.com/drive/folders/1WI-oqDQz73j28oMcFe2h_HxMv-oN9kH9?usp=sharing) [:clipboard:](./Baseline-tiered_imagenet-Conv64F-Table2.yaml) | Table2 |
| 2 | ResNet18 | - | 64.65 ± 0.41 [:arrow_down:](https://drive.google.com/drive/folders/12YoN2Xd6tAlJQ-Yd5KZyXqubo1q0B11S?usp=sharing) [:clipboard:](./Baseline-tiered_imagenet-resnet18-Table2.yaml) | - | 82.73 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/12YoN2Xd6tAlJQ-Yd5KZyXqubo1q0B11S?usp=sharing) [:clipboard:](./Baseline-tiered_imagenet-resnet18-Table2.yaml) | Table2 |
