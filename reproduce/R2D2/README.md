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
| 3 | ResNet18 | - | 58.36 ± 0.38 [:arrow_down:](https://drive.google.com/drive/folders/1ni8MselKFMex9e9Ck9ehsUdNMoM5i220?usp=sharing) [:clipboard:](./R2D2-miniImageNet--ravi-resnet18-5-1-Table2.yaml) | - | 75.69 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1jVcT8J6pZXsza5nhM9jtxg0msllmfnvo?usp=sharing) [:clipboard:](./R2D2-miniImageNet--ravi-resnet18-5-5-Table2) | Table.2 |

|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 52.18 ± 0.40 [:arrow_down:](https://drive.google.com/drive/folders/1G0NS_pmLl6nMiDxkeUI46dCLMI0yRRzE?usp=sharing) [:clipboard:](./R2D2-tiered_imagenet-Conv64F-5-1-Table2.yaml) | - | 69.19 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/1yTpBxVZsp13Qs45Oxodv7zjx2HcpNIuQ?usp=sharing) [:clipboard:](./R2D2-tiered_imagenet-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 65.07 ± 0.44 [:arrow_down:](https://drive.google.com/drive/folders/1ph9_NC04tNqHwpagG9ggFPxlzH5kCd6W?usp=sharing) [:clipboard:](./R2D2-tiered_imagenet-resnet12-5-1-Table2.yaml) | - | 83.04 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1S59uh1zRNobVnipgYBczfFq60541w19R?usp=sharing) [:clipboard:](./R2D2-tiered_imagenet-resnet12-5-5-Table2.yaml) | Table.2 |
| 3 | ResNet18 | - | 64.73 ± 0.44 [:arrow_down:](https://drive.google.com/drive/folders/1IMCcasMboW7Q0DuGDu7q_kQO3kV4cmGR?usp=sharing) [:clipboard:](./R2D2-tiered_imagenet-resnet18-5-1-Table2.yaml) | - | 83.40 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/157kAoBfa12JzoaHUszwd8bqVO-TCkMy0?usp=sharing) [:clipboard:](./R2D2-tiered_imagenet-resnet18-5-5-Table2.yaml) | Table.2 |
