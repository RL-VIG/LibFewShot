# Mutual Centralized Learning for Few-shot Classification

## Introduction

| **Name:**   | [MCL](https://arxiv.org/abs/2106.05517)                      |
| ----------- | ------------------------------------------------------------ |
| **Embed.:** | Conv64F/ResNet12                                             |
| **Type:**   | Metric                                                       |
| **Venue:**  | CVPR’22                                                      |
| **Codes:**  | [Mutual Centralized Learning](https://github.com/LouieYang/MCL) |

Cite this work with:

```bibtex
@InProceedings{Liu_2022_CVPR,
  author    = {Liu, Yang and Zhang, Weifeng and Xiang, Chao and Zheng, Tu and Cai, Deng},
  title     = {Learning To Affiliate: Mutual Centralized Learning for Few-Shot Classification},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  pages     = {14411-14420}
}
```

---

## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | *55.55 | 55.82 [:arrow_down:](https://pan.baidu.com/s/1IkqbBoE4oQ7k3JCtirmwWQ?pwd=2xyh) code: 2xyh [:clipboard:]() | *71.74 | 71.09  [:arrow_down:](https://pan.baidu.com/s/1zAaCXfqR0zzinnBTbh-w0w?pwd=mp22) code: mp22 [:clipboard:]() | Table.2 |
| 2 | ResNet12 | *67.51 | 69.16 [:arrow_down:](https://pan.baidu.com/s/1gN4pqj4sz-zvEc5BBO5e7w?pwd=y0gz) code: y0gz [:clipboard:]() | *83.99 | 83.28 [:arrow_down:](https://pan.baidu.com/s/1zFEvp7ttIVWssuo7Gk4D7A?pwd=ehu1) code: ehu1 [:clipboard:]() | Table.2 |


|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | R2D2 | - | 64.67 [:arrow_down:](https://pan.baidu.com/s/1Yty8fVytO3GzpYHEk5qolQ?pwd=mq57) code: mq57 [:clipboard:]() | - | 81.92  [:arrow_down:](https://pan.baidu.com/s/1WRDtkuOL8cxAyJcrhjsMMA?pwd=abxr) code: abxr [:clipboard:]() | Table.2 |
| 2 | R2D2+MCL | - | 66.03 [:arrow_down:](https://pan.baidu.com/s/11cZX0lVWi47M0rXphNUxKg?pwd=kvzq) code: kvzq [:clipboard:]() | - | 82.71 [:arrow_down:](https://pan.baidu.com/s/1rHtMrSNWI-AcY8KMhe38Cw?pwd=533v) code: 533v [:clipboard:]() | Table.2 |

It is worth noting that the classifier MCL can directly utilize the existing backbone in the framework. The use of the additionally defined `conv_four_mcl.py` and `resnet12_12_mcl.py` files serves as control variables for comparing accuracy with the original paper. The weights for `resnet12` used in `resnet12_12_mcl.py` can be downloaded here[:arrow_down:](https://pan.baidu.com/s/1TetYehPJuD-iDo7wrBP5Qw?pwd=ndhu) (code: ndhu).