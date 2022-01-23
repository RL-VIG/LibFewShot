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
| 1 | Conv64F | *49.61 ± 0.16 | 47.92 ± 0.35 [:arrow_down:](https://drive.google.com/drive/folders/13BhKmNtGgETLOoOCrhhGEFqGGWIE27_U?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-Conv64F-5-1-Table2.yaml) | *66.45 ± 0.37 | 64.39 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1ynVJ91zzs7Lw31oo7OeX5rO2BIyg6r2T?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 58.87 ± 0.38 [:arrow_down:](https://drive.google.com/drive/folders/17edwZpd0WC3E6HQrb6tuDFF9vkX9Qict?usp=sharing) [:clipboard:](BOIL-miniImageNet--ravi-resnet12-5-1-Table2) | - | 72.88 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1_hwL858a1KptqS7a2BYL-UG_8jQoLxnH?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-resnet12-5-5-Table2.yaml) | Table.2 |
| 3 | ResNet18 | - | 57.85 ± 0.40 [:arrow_down:](https://drive.google.com/drive/folders/1GTWErz-jzkUNTIR_zfguFCdASADsKJRO?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-resnet18-5-1-Table2.yaml) | - | 70.84 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1VDwzcVOMSVg8qg2mX8bzIjJwURbbjMjw?usp=sharing) [:clipboard:](BOIL-miniImageNet--ravi-resnet18-5-5-Table2.yaml) | Table.2 |
<!-- | 1 | Conv64F | 49.61 ± 0.16 | 48.00 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/18rH2HgKtVnEETfb8XcUkFtPmWQJeHjFB?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-Conv64F-5-1.yaml) | 66.45 ± 0.37 | 59.70 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1YjXpzYOZud60tbbiYNhviAN9Rpx-5LfJ?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-Conv64F-5-5-Reproduce.yaml) | Once_update |
| 2 | ResNet12woLSC | - | 52.75 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1Of1WK7K4x732GRzsWTPAu9DCiEQCOqRK?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-resnet12woLSC-5-1.yaml) | 71.30 ± 0.28 | 61.67 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1KOlzMq_ggCH5fse-Vx4q_HUkA8yV5VMr?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-resnet12woLSC-5-5-Reproduce.yaml) | Once_update |
| 3 | ResNet12 | - |  | - | 67.41 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1_hwL858a1KptqS7a2BYL-UG_8jQoLxnH?usp=sharing) [:clipboard:](./BOIL-miniImageNet--ravi-resnet12-5-5-Reproduce.yaml) | Once_update | -->



|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | *48.58 ± 0.27 | 50.04 ± 0.38 [:arrow_down:](https://drive.google.com/drive/folders/1LHdhlXJgvBKnpqNH1F1wtbT-20j3yxGz?usp=sharing) [:clipboard:](./BOIL-tiered_imagenet-Conv64F-5-1-Table2.yaml) | *69.37 ± 0.12 | 65.51 ± 0.34 [:arrow_down:](https://drive.google.com/drive/folders/1kwcnb2KDFNmeNZqA-KLH-X1QHfq6PlVt?usp=sharing) [:clipboard:](./BOIL-tiered_imagenet-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 64.66 ± 0.44 [:arrow_down:](https://drive.google.com/drive/folders/1EQUh-RvacF-8f2Qn-W9Z4DDQTfILmVwd?usp=sharing) [:clipboard:](./BOIL-tiered_imagenet-resnet12-5-1-Table2.yaml) | - | 80.38 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1mZt7XN5gQI-dt3FVsrTTMKu0mSKhTw3e?usp=sharing) [:clipboard:](./BOIL-tiered_imagenet-resnet12-5-5-Table2.yaml) | Table2 |
| 3 | ResNet18 | - | 60.85 ± 0.45
 [:arrow_down:](https://drive.google.com/drive/folders/1Twd8eJKUa3lrY7aDnKHqM3sUmMLX05sg?usp=sharing) [:clipboard:](./BOIL-tiered_imagenet-resnet18-5-1-Table2.yaml) | - | 77.74 ± 0.38 [:arrow_down:](https://drive.google.com/drive/folders/1WrjMtYUDQ7P76op4BGO4ctn8gRVBPqie?usp=sharing) [:clipboard:](./BOIL-tiered_imagenet-resnet18-5-5-Table2.yaml) | Table.2 |
<!-- | 2 | ResNet12woLSC | - | - | 73.44 | - | Once_update | -->
<!-- | 1 | Conv64F |  | - |  | - | Once_update | -->
