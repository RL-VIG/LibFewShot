# A Closer Look at Few-shot Classification
## Introduction
| Name:    | [Baseline](https://arxiv.org/abs/1904.04232)                          |
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
| 1 | Conv64F | 42.11 | 42.34 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1GKt_Y-CZqgzsm4YQEkeP0j_xTvjKhlng?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-Conv64F-5-Reproduce.yaml) | 62.53 | 62.18 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1GKt_Y-CZqgzsm4YQEkeP0j_xTvjKhlng?usp=sharing) [:clipboard:](./Baseline-miniImageNet--ravi-Conv64F-5-Reproduce.yaml) | Table2 |



<!-- |   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | - [:arrow_down:]() [:clipboard:]() | - | - [:arrow_down:]() [:clipboard:]() | Table2 | -->
