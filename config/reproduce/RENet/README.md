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

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *miniI*mageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12 | 67.60 ± 0.44 | 66.83 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/1vU3vprzjwxLTa9wgzi5HPXughkmxhMHS?usp=sharing) [:clipboard:](./RENet-miniImageNet--ravi-resnet12-5-1-Reproduce.yaml)| 82.58 ± 0.30 | 82.13 ± 0.26 [:arrow_down:](https://drive.google.com/drive/folders/1JH1olkacMQdZTUq3zTHX_55KL48Fzloi?usp=sharing) [:clipboard:](./RENet-miniImageNet--ravi-resnet12-5-5-Reproduce.yaml)| Reproduce |
| 2 | Conv64F | - | 57.62 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/12SuX47W1hJTWuxUGu588ZcuGHpta34lE?usp=sharing) [:clipboard:](./RENet-miniImageNet--ravi-Conv64F-5-1-Table2.yaml) | - | 74.14 ± 0.27 [:arrow_down:](https://drive.google.com/drive/folders/1NUYKt1-BhK4TkwOr0kRCMuWhGw-02tCx?usp=sharing) [:clipboard:](./RENet-miniImageNet--ravi-Conv64F-5-5-Table2.yaml) | Table.2 |
| 3 | ResNet12 | - | 64.81 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/12cH2D3KL3bIi7Fh1AssAP7pSOACvJXq8?usp=sharing) [:clipboard:](./RENet-miniImageNet--ravi-resnet12-5-1-Table2.yaml) | - |  79.90 ± 0.27 [:arrow_down:](https://drive.google.com/drive/folders/1FVkLNX8n8OLXUpu8M7-3F47Th8K6Z4-P?usp=sharing) [:clipboard:](./RENet-miniImageNet--ravi-resnet12-5-5-Table2.yaml) | Table.2 |
| 3 | ResNet18 | - | 62.86 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1otOFnwYDBgQuy-Lc49-yrarT4djxd5SL?usp=sharing) [:clipboard:](./RENet-miniImageNet--ravi-resnet18-5-1-Table2.yaml) | - | - | Table.2 |

|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12 | 71.61 ± 0.5 | 71.23 ± 0.42 [:arrow_down:](https://drive.google.com/drive/folders/1DKt-3Bs6u7hDorxWoRLQbdklQVwqDNew?usp=sharing) [:clipboard:](./RENet-tiered_imagenet-resnet12-5-1-Reproduce.yaml) | 85.28 ± 0.35 | 84.78 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/11n6Lk2vPJPYBwGFOh2WY8EKzC4gSwOEE?usp=sharing) [:clipboard:](./RENet-tiered_imagenet-resnet12-5-5-Reproduce.yaml) | Reproduce |
| 2 | Conv64F | - | 61.62 ± 0.40 [:arrow_down:](https://drive.google.com/drive/folders/1SRmd0KGq92Ppwr1oPlvOeKO8CuQ6ClHz?usp=sharing) [:clipboard:](./RENet-tiered_imagenet-Conv64F-5-1-Table2.yaml) | - | 76.74 ± 0.33[:arrow_down:](https://drive.google.com/drive/folders/1JNcBzGqPbEgKSUjDq-LIh9clOo7SBj4Q?usp=sharing) [:clipboard:](./RENet-tiered_imagenet-Conv64F-5-5-Table2.yaml) | Table.2 |
| 3 | ResNet12 | - | 70.14 ± 0.43 [:arrow_down:](https://drive.google.com/drive/folders/1R7AQOUaAby-bCjOAFxqfwCRIxWJEaJqo?usp=sharing) [:clipboard:](./RENet-tiered_imagenet-resnet12-5-1-Table2.yaml) | - | 82.70 ± 0.31 [:arrow_down:](https://drive.google.com/drive/folders/1OrV5GyF4HR-jJ57KV0MpvdBj1J-WbAay?usp=sharing) [:clipboard:](./RENet-tiered_imagenet-resnet12-5-5-Table2.yaml) | Table.2 |
| 4 | ResNet18 | - | 71.53 ± 0.43 [:arrow_down:](https://drive.google.com/drive/folders/1V7Mhj8e3K6LbTUdtkAL5NmyS_EonFnrd?usp=sharing) [:clipboard:](./RENet-tiered_imagenet-resnet18-5-1-Table2.yaml) | - | - [:arrow_down:]() [:clipboard:]() | Table.2 |
