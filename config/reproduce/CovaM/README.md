# Distribution Consistency Based Covariance Metric Networks for Few-Shot Learning
## Introduction
| Name:    | [CovaM](https://doi.org/10.1609/aaai.v33i01.33018642)  |
|----------|-------------------------------|
| Embed.:  | ResNet12 |
| Type:    | Metric       |
| Venue:   | AAAI'19                      |
| Codes:   | [**CovaM**](https://github.com/WenbinLee/CovaMNet)  |

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/aaai/LiXHWGL19,
  author    = {Wenbin Li and
               Jinglin Xu and
               Jing Huo and
               Lei Wang and
               Yang Gao and
               Jiebo Luo},
  title     = {Distribution Consistency Based Covariance Metric Networks for Few-Shot
               Learning},
  booktitle = {The Thirty-Third {AAAI} Conference on Artificial Intelligence, {AAAI}
               2019, The Thirty-First Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2019, The Ninth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2019, Honolulu, Hawaii,
               USA, January 27 - February 1, 2019},
  pages     = {8642--8649},
  year      = {2019},
  url       = {https://doi.org/10.1609/aaai.v33i01.33018642},
  doi       = {10.1609/aaai.v33i01.33018642}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *miniI*mageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 51.59 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/1yJ-jqQNACIYITM8fK-g7_bG8nYFk86H9?usp=sharing) [:clipboard:](./ConvMNet-miniImageNet--ravi-Conv64F-5-1-Table2.yaml) | - | 67.65 ± 0.32 [:arrow_down:](https://drive.google.com/drive/folders/10moUkNzdrxtDqM15_EtmZVyQ_Czg7jhK?usp=sharing) [:clipboard:](./ConvMNet-miniImageNet--ravi-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 56.95 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1hg2Fd1iByx2nqQfNPUyeaNYU3ng5BbEF?usp=sharing) [:clipboard:](./ConvMNet-miniImageNet--ravi-resnet12-5-1-Table2.yaml) | - | 71.41 ± 0.66 [:arrow_down:](https://drive.google.com/drive/folders/1wiAfGuLITFl8QsS42fRIc9SvIco5rsfb?usp=sharing) [:clipboard:](./ConvMNet-miniImageNet--ravi-resnet12-5-5-Table2.yaml)| Table.2 |
| 3 | ResNet18 | - | 55.83 ± 0.39 [:arrow_down:](https://drive.google.com/drive/folders/1dFrq1QiV3lD9B8O4VYj2t51uAlyYCoYB?usp=sharing) [:clipboard:](./ConvMNet-miniImageNet--ravi-resnet18-5-1-Table2.yaml)| - | 70.97 ± 0.33 [:arrow_down:](https://drive.google.com/drive/folders/1a-4jSXWPCmxr0uTDBTYcM9OGLlEUAMfb?usp=sharing) [:clipboard:](./ConvMNet-miniImageNet--ravi-resnet18-5-5-Table2.yaml) | Table.2 |

|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 51.92 ± 0.40 [:arrow_down:](https://drive.google.com/drive/folders/1CoUXTk9rfWHqmdNM3zAl4iYORe8q6uzR?usp=sharing) [:clipboard:](./ConvMNet-tiered_imagenet-Conv64F-5-1-Table2.yaml) | - | 69.76 ± 0.34 [:arrow_down:](https://drive.google.com/drive/folders/12GDfn9uvLh6bS_BSe5QAoLGRg8zKuEf1?usp=sharing) [:clipboard:](./ConvMNet-tiered_imagenet-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 58.49 ± 0.42 [:arrow_down:](https://drive.google.com/drive/folders/1AWj0dypTaOVssQ81wZSMGgVEEP4mpSQe?usp=sharing) [:clipboard:](./ConvMNet-tiered_imagenet-resnet12-5-1-Table2.yaml) | - | 76.34 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1tTMPzb8NbcrjGONuFD8NM-rFifUnH-nS?usp=sharing) [:clipboard:](./ConvMNet-tiered_imagenet-resnet12-5-5-Table2.yaml)| Table.2 |
| 3 | ResNet18 | - | 54.12 ± 0.47 [:arrow_down:](https://drive.google.com/drive/folders/1DU9fa2ZwNYYAwNJMfVNADC31UzvSBZcR?usp=sharing) [:clipboard:](./ConvMNet-tiered_imagenet-resnet18-5-1-Table2.yaml)| - | 73.51 ± 0.55 [:arrow_down:](https://drive.google.com/drive/folders/1GQUYOBDd5L6UDAMPd4b5jk0F4tjX0pRr?usp=sharing) [:clipboard:](./ConvMNet-tiered_imagenet-resnet18-5-5-Table2.yaml) | Table.2 |
