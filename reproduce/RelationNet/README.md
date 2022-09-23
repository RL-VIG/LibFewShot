# Learning to Compare: Relation Network for Few-Shot Learning
## Introduction
| Name:    | [Relation Network](https://arxiv.org/abs/1703.05175)  |
|----------|-------------------------------|
| Embed.:  | Conv64F |
| Type:    | Metric       |
| Venue:   | CVPR'18                      |
| Codes:   | [**LearningToCompare_FSL**](https://github.com/floodsung/LearningToCompare_FSL) |

+ We use ResNet-18 with `last_stride = 1` for experiments of Table.2.

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/cvpr/SungYZXTH18,
  author    = {Flood Sung and
               Yongxin Yang and
               Li Zhang and
               Tao Xiang and
               Philip H. S. Torr and
               Timothy M. Hospedales},
  title     = {Learning to Compare: Relation Network for Few-Shot Learning},
  booktitle = {2018 {IEEE} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2018, Salt Lake City, UT, USA, June 18-22, 2018},
  pages     = {1199--1208},
  year      = {2018},
  url       = {http://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html},
  doi       = {10.1109/CVPR.2018.00131}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 51.52 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/15mzj4uTZ9XlddKFduUACwtBeU51FeQxI?usp=sharing) [:clipboard:](./RelationNet-miniImageNet--ravi-Conv64F-5-1-Table2.yaml) | - | 66.49 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1kIz-Zgok60kqbygfsmaKgkU9_ZxBGEGH?usp=sharing) [:clipboard:](./RelationNet-miniImageNet--ravi-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 55.22 ± 0.39 [:arrow_down:](https://drive.google.com/drive/folders/1Z9rfIkKYLd9vFbFj4-qa87FzAu0REPoh?usp=sharing) [:clipboard:](./RelationNet-miniImageNet--ravi-resnet12-5-1-Table2.yaml) | - | 69.25 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1KgwJ7zulL1byciS1CF4CRDYq2i-qoH0W?usp=sharing) [:clipboard:](./RelationNet-miniImageNet--ravi-resnet12-5-5-Table2.yaml) | Table.2 |
| 3 | ResNet18 | - | 53.98 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1qrslqQOpg0Qa5dmmeMOtVSny3tYmmUFH?usp=sharing) [:clipboard:](./RelationNet-miniImageNet--ravi-resnet18-5-1-Table2.yaml) | - | 71.27 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1idr5_wKw8zdPNuXg8-iCCknHSEEwOXfD?usp=sharing) [:clipboard:](./RelationNet-miniImageNet--ravi-resnet18-5-5-Table2.yaml) | Table.2 |


|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 54.37 ± 0.44 [:arrow_down:](https://drive.google.com/drive/folders/1S_Zx8ptBUyzzz9ZoQfRofInuxKTRKhFm?usp=sharing) [:clipboard:](./RelationNet-tiered_imagenet-Conv64F-5-1-Table2.yaml) | - | 71.93 ± 0.35 [:arrow_down:](https://drive.google.com/drive/folders/1VXUiDAZXrsbb2FgVeRwODEBQ7oat1CLL?usp=sharing) [:clipboard:](./RelationNet-tiered_imagenet-Conv64F-5-5-Table2.yaml) | Table.2 |
