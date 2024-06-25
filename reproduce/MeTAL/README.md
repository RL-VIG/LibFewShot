# Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning
## Introduction
| Name:    | [MeTAL](https://arxiv.org/abs/2110.03909) |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ |
| Type:    | Meta       |
| Venue:   | ICCV'21                      |
| Codes:   | [**MeTAL**](https://github.com/baiksung/MeTAL) |


Cite this work with:
```bibtex
@inproceedings{baik2021meta,
  title={Meta-learning with task-adaptive loss function for few-shot learning},
  author={Baik, Sungyong and Choi, Janghoon and Kim, Heewon and Cho, Dohee and Min, Jaesik and Lee, Kyoung Mu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={9465--9474},
  year={2021}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | 52.63 ± 0.37% | 53.753 [:arrow_down:](https://drive.google.com/drive/folders/1LbL4BGtdypVri6upeMwjpgQKIxW5u2Zg?usp=share_link) [:clipboard:](./MeTAL-miniImageNet--ravi-Conv64F-5-1.yaml) | 70.52 ± 0.29% | 71.233 [:arrow_down:](https://drive.google.com/drive/folders/1nzQuPq8wkUwhLZMyd3675r-COj4_TRaS?usp=sharing) [:clipboard:](./MeTAL-miniImageNet--ravi-Conv64F-5-5.yaml) | Comments |
| 2 | ResNet12 | 59.64 ± 0.38% | 60.333 [:arrow_down:](https://drive.google.com/drive/folders/160eboS9b6L0HiFzO7Y_02o84MnxiDNEM?usp=sharing) [:clipboard:](./MeTAL-miniImageNet--ravi-resnet12-5-1.yaml) | 76.20 ± 0.19% | 76.800 [:arrow_down:](https://drive.google.com/drive/folders/1571BXod4mq7PWb8WD_FglaLoOZbKfaNl?usp=share_link) [:clipboard:](./MeTAL-miniImageNet--ravi-resnet12-5-5.yaml) | Comments |
