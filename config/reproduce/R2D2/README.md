# Meta-learning with differentiable closed-form solvers
## Introduction
| Name:    | [R2D2](https://arxiv.org/abs/1805.08136)  |
|----------|-------------------------------|
| Embed.:  | Conv64F |
| Type:    | Metric       |
| Venue:   | ICLR'19                      |
| Codes:   | [*MetaOptNet*](https://github.com/kjunelee/MetaOptNet) |

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/iclr/BertinettoHTV19,
  author    = {Luca Bertinetto and
               Jo{\\~{a}}o F. Henriques and
               Philip H. S. Torr and
               Andrea Vedaldi},
  title     = {Meta-learning with differentiable closed-form solvers},
  booktitle = {7th International Conference on Learning Representations, {ICLR} 2019,
               New Orleans, LA, USA, May 6-9, 2019},
  year      = {2019},
  url       = {https://openreview.net/forum?id=HyxnZh0ct7}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 51.19 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/1GbpOJD4mJz5e-hNxAKh-vQISm3P6OykG?usp=sharing) [:clipboard:](./R2D2-miniImageNet--ravi-Conv64F-5-1-Table2.yaml) | - | 67.29 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1Oag2uuGInA8eDXDDlCzOpWwVak4pXjuS?usp=sharing) [:clipboard:](./R2D2-miniImageNet--ravi-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 59.52 ± 0.39 [:arrow_down:](https://drive.google.com/drive/folders/1ZegHyAoZvBBMAd6UAt-D2Ul_lwods8Vg?usp=sharing) [:clipboard:](./R2D2-miniImageNet--ravi-resnet12-5-1-Table2.yaml) | - | 74.61 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1-S-C2nVDp0JVghq1MVTefiUquPVBOPN7?usp=sharing) [:clipboard:](./R2D2-miniImageNet--ravi-resnet12-5-5-Table2.yaml) | Table.2 |
