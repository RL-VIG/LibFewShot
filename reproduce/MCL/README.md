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

|      | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1)                              | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5)                              | :memo: Comments |
| ---- | --------- | --------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ | --------------- |
| 1    | Conv64F   | *55.55                      | 55.35 ± 0.32 [:arrow_down:](https://pan.baidu.com/s/1TgZzrPN-5uFBWy1hTuH4Kw?pwd=q2ae) code: q2ae [:clipboard:]() | *71.74                     | 71.67 ± 0.27 [:arrow_down:](https://pan.baidu.com/s/1Wn6NMgK-MRrxhKxEOSwe_Q?pwd=81fi) code: 81fi [:clipboard:]() | Table.2         |
| 2    | ResNet12  | *67.51                      | 67.11 ± 0.30 [:arrow_down:](https://pan.baidu.com/s/1TjIDXKEafRw3amh_HEwlww?pwd=mang) code: mang [:clipboard:]() | *83.99                     | 83.41 ± 0.28 [:arrow_down:]( https://pan.baidu.com/s/11dQm91mvllmA_1aOZdePUw?pwd=7sjm) code: 7sjm [:clipboard:]() | Table.2         |

|      | Embedding | :book:*tiered*ImageNet(5,1) | :computer:*tiered*ImageNet(5,1)                              | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet(5,5)                             | :memo: Comments |
| ---- | --------- | --------------------------- | ------------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------ | --------------- |
| 1    | Conv64F   | *57.78                      | 56.82 ± 0.22 [:arrow_down:](https://pan.baidu.com/s/1hy2Q4kl9wA3ZfX8QecHVGg?pwd=t26r) code: t26r [:clipboard:]() | *74.77                       | 73.48 ± 0.17 [:arrow_down:](https://pan.baidu.com/s/1CGCOFGXC0Y_emuixDs_wrA?pwd=6xjd) code: 6xjd [:clipboard:]() | Table.2         |
| 2    | Resnet12  | *72.01                      | 71.23 ± 0.15 [:arrow_down:](https://pan.baidu.com/s/1Cmz-THbBnto6wsETNeMncw?pwd=3v8i) code: 3v8i [:clipboard:]() | *86.02                       | 85.24 ± 0.21 [:arrow_down:](https://pan.baidu.com/s/1ALQL6D5jQOuu8_3JIhgk1A?pwd=3mwf) code: 3mwf [:clipboard:]() | Table.2         |

**Evaluate methods with the pretrained backbone ResNet-12 without meta-training**

|      | Classifier | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1)                              | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5)                              | :memo: Comments |
| ---- | --------- | --------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ | --------------- |
| 1    | R2D2      | -                           | 60.48 [:arrow_down:](https://pan.baidu.com/s/1t2JUmVhCQMxAftnTtwHCPQ?pwd=pgz2) code: pgz2 [:clipboard:]() | -                          | 78.96 [:arrow_down:](https://pan.baidu.com/s/1wX0WteYGWfouNNBjbVS2Cw?pwd=jfh8) code: jfh8 [:clipboard:]() | Table.2         |
| 2    | R2D2+MCL  | -                           | 61.29 $\textcolor{red}{(+0.81)}$ [:arrow_down:](https://pan.baidu.com/s/1CQM9B0uetwZNeQdGOCLt9Q?pwd=hysq) code: hysq [:clipboard:]() | -                          | 79.92 $\textcolor{red}{(+0.96)}$ [:arrow_down:](https://pan.baidu.com/s/1Tf22KF56wXTtyqzTNMUFdA?pwd=y4qj) code:  y4qj [:clipboard:]() | Table.2         |

|      | Classifier | :book:*tiered*ImageNet(5,1) | :computer:*tiered*ImageNet(5,1)                              | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet(5,5)                             | :memo: Comments |
| ---- | --------- | --------------------------- | ------------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------ | --------------- |
| 1    | R2D2      | -                           | 69.63  [:arrow_down:](https://pan.baidu.com/s/1LALA2YDQsHo3iYsqJTHODw?pwd=psuc) code: psuc [:clipboard:]() | -                            | 84.82 [:arrow_down:](https://pan.baidu.com/s/1WZHNcfoKyrWhwaUg8xlcjg?pwd=s3rc) code: s3rc [:clipboard:]() | Table.2         |
| 2    | R2D2+MCL  | -                           | 70.21 $\textcolor{red}{(+0.58)}$ [:arrow_down:](https://pan.baidu.com/s/10devDYuSLZGDB_xkHLwyDw?pwd=34ft) code: 34ft [:clipboard:]() | -                            | 85.31 $\textcolor{red}{(+0.49)}$ [:arrow_down:](https://pan.baidu.com/s/10YcIzQtXZILCKrkl4Zo7yg?pwd=kr6p) code: kr6p [:clipboard:]() | Table.2         |

It is worth noting that the classifier MCL can directly utilize the existing backbone in the framework. The use of the additionally defined `conv_four_mcl.py` and `resnet12_12_mcl.py` files serves as control variables for comparing accuracy with the original paper. The weights for `resnet12` used in `resnet12_12_mcl.py` can be downloaded here[:arrow_down:](https://pan.baidu.com/s/12dpz9dHZk_bB0o7Hc11Ifg?pwd=6npc) (code: 6npc). 
