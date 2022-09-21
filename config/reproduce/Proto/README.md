# Prototypical Networks for Few-shot Learning
## Introduction
| Name:    | [Prototypical Networks](https://arxiv.org/abs/1703.05175)  |
|----------|-------------------------------|
| Embed.:  | Conv64F |
| Type:    | Metric       |
| Venue:   | NeurIPS'17                      |
| Codes:   | [*Prototypical-Networks-for-Few-shot-Learning-PyTorch*](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch) |

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/nips/SnellSZ17,
  author    = {Jake Snell and
               Kevin Swersky and
               Richard S. Zemel},
  title     = {Prototypical Networks for Few-shot Learning},
  booktitle = {Advances in Neural Information Processing Systems 30: Annual Conference
               on Neural Information Processing Systems 2017, December 4-9, 2017,
               Long Beach, CA, {USA}},
  pages     = {4077--4087},
  year      = {2017},
  url       = {https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 47.05 ± 0.35 [:arrow_down:](https://drive.google.com/drive/folders/1OjobWtwiGbH9kkI7Zzh2tg5Y0Eh8O3zM?usp=sharing) [:clipboard:](./ProtoNet-miniImageNet-Conv64F-5-1-Table2.yaml) | - | 68.56 ± 0.16 [:arrow_down:](https://drive.google.com/drive/folders/1kekt2wiecx4TVgKiDCAiXM-cBfrQ3YC3?usp=sharing) [:clipboard:](./ProtoNet-miniImageNet-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 54.25 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1N1BjE8yl6f1Hz9LcgzuvND4sawQebQef?usp=sharing) [:clipboard:](./ProtoNet-miniImageNet--ravi-resnet12-5-1-Table2.yaml) | - | 74.65 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1fxPcT_QYI2dYAH0rR-bApeq4XmRnd3WY?usp=sharing) [:clipboard:](./ProtoNet-miniImageNet--ravi-resnet12-5-5-Table2.yaml) | Table.2 |



|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 46.11 ± 0.39 [:arrow_down:](https://drive.google.com/drive/folders/1GSs680y1hMixEbz6HWCZjRqs7SxKIQEF?usp=sharing) [:clipboard:](./ProtoNet-tiered_imagenet-Conv64F-5-1-Table2.yaml) | - | 70.07 ± 0.34 [:arrow_down:](https://drive.google.com/drive/folders/1AjeIFPB2K5W93lYrU2iulTX2MKxTogd_?usp=sharing) [:clipboard:](./ProtoNet-tiered_imagenet-Conv64F-5-5-Table2.yaml) | Table.2 |
