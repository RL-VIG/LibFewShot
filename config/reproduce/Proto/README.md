# Prototypical Networks for Few-shot Learning
## Introduction
| Name:    | [Prototypical Networks](https://arxiv.org/abs/1703.05175)  |
|----------|-------------------------------|
| Embed.:  | Conv64F |
| Type:    | Metric       |
| Venue:   | NeurIPS'17                      |
| Codes:   | [*Prototypical-Networks-for-Few-shot-Learning-PyTorch*]((https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch))                   |

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

|   | Config | Embedding | miniImageNet (5,1) | miniImageNet (5,5) |
|---|---| -----------|--------------------|--------------------|
| 1 | Table.2 | Conv64F   | [47.050 ± 0.354](https://drive.google.com/drive/folders/1OjobWtwiGbH9kkI7Zzh2tg5Y0Eh8O3zM?usp=sharing) [YAML]((./ProtoNet-miniImageNet-Conv64F-5-1-Table2.yaml))       | [68.564 ± 0.164](https://drive.google.com/drive/folders/1kekt2wiecx4TVgKiDCAiXM-cBfrQ3YC3?usp=sharing) [YAML](./ProtoNet-miniImageNet-Conv64F-5-5-Table2.yaml)  |
