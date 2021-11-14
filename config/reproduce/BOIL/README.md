# BOIL: Towards Representation Change for Few-shot Learning
## Introduction
| Name:    | [BOIL](https://openreview.net/forum?id=umIdUL8rMH)                          |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ResNet18 |
| Type:    | Meta       |
| Venue:   | ICLR'21                      |
| Codes:   | [**BOIL**](https://github.com/HJ-Yoo/BOIL)|

:bangbang: The authors of BOIL didn't release the train args except for CIFAR-FS. We can't reproduce by the hyper-parameters they claimed. So we choose our hyper-parameters.

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/iclr/OhYKY21,
  author    = {Jaehoon Oh and
               Hyungjun Yoo and
               ChangHwan Kim and
               Se{-}Young Yun},
  title     = {{BOIL:} Towards Representation Change for Few-shot Learning},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=umIdUL8rMH},
}

```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | 49.61 ± 0.16 | 48.00 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/18rH2HgKtVnEETfb8XcUkFtPmWQJeHjFB?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-Conv64F-5-1.yaml) | 66.45 ± 0.37 | - | Once_update |
| 2| ResNet12woLSC | - | 52.75 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1Of1WK7K4x732GRzsWTPAu9DCiEQCOqRK?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-resnet12woLSC-5-1.yaml) | 71.30 ± 0.28 | - | Once_update |

|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | 48.58 ± 0.27 | - | 69.37 ± 0.12 | - | Once_update |
| 2 | ResNet12woLSC | - | - | 73.44 | - | Once_update |
