# Relational Embedding for Few-Shot Classification
## Introduction
| Name:    | [RENet](https://arxiv.org/abs/2108.09666)  |
|----------|-------------------------------|
| Embed.:  | ResNet12 |
| Type:    | Fine-tuning       |
| Venue:   | ICCV'21                      |
| Codes:   | [**renet**](https://github.com/dahyun-kang/renet/)  |

Cite this work with:
```bibtex
@article{DBLP:journals/corr/abs-2108-09666,
  author    = {Dahyun Kang and
               Heeseung Kwon and
               Juhong Min and
               Minsu Cho},
  title     = {Relational Embedding for Few-Shot Classification},
  journal   = {CoRR},
  volume    = {abs/2108.09666},
  year      = {2021},
}

```
---
## Results and Models

**Classification**

|   | Embedding | :book: miniImageNet (5,1) | :computer: miniImageNet (5,1) | :book:miniImageNet (5,5) | :computer: miniImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12 | 67.60 ± 0.44 |  66.83 ± 0.36[:arrow_down:](https://drive.google.com/drive/folders/1vU3vprzjwxLTa9wgzi5HPXughkmxhMHS?usp=sharing) [:clipboard:](./RENet-miniImageNet--ravi-resnet12-5-1-Reproduce.yaml)| 82.58 ± 0.30| - | Reproduce |
