# Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning

## Introduction

| Name:   | [MeTAL](https://arxiv.org/abs/2110.03909)      |
| ------- | ---------------------------------------------- |
| Embed.: | Conv64F/ResNet12/                              |
| Type:   | Meta                                           |
| Venue:  | ICCV'21                                        |
| Codes:  | [**MeTAL**](https://github.com/baiksung/MeTAL) |


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

|      | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1)                              | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5)                              | :memo: Comments |
| ---- | --------- | --------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ | --------------- |
| 1    | Conv64F   | 52.63 ± 0.37%               | 55.48 ± 0.40% [:arrow_down:](https://drive.google.com/drive/folders/13EA9LJIDKz26cUQ_2DFOXoX6I9jU5X8e?usp=sharing) [:clipboard:](./METAL-miniImageNet--ravi-Conv64F-5-1.yaml) | 70.52 ± 0.29%              | 70.89 ± 0.30% [:arrow_down:](https://drive.google.com/drive/folders/1nzQuPq8wkUwhLZMyd3675r-COj4_TRaS?usp=share_link) [:clipboard:](./METAL-miniImageNet--ravi-Conv64F-5-5.yaml) |                 |
| 2    | ResNet12  | 59.64 ± 0.38%               | 59.23 ± 0.57% [:arrow_down:](https://drive.google.com/drive/folders/160eboS9b6L0HiFzO7Y_02o84MnxiDNEM?usp=share_link) [:clipboard:](./METAL-miniImageNet--ravi-resnet12-5-1.yaml) | 76.20 ± 0.19%              | 77.84 ± 0.45% [:arrow_down:](https://drive.google.com/drive/folders/1571BXod4mq7PWb8WD_FglaLoOZbKfaNl?usp=share_link) [:clipboard:](./METAL-miniImageNet--ravi-resnet12-5-5.yaml) |                 |

|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
| ---- | --------- | --------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ | --------------- |
| 1    | Conv64F   | 54.34 ± 0.31%          | 55.60 ± 0.44% [:arrow_down:](https://drive.google.com/drive/folders/1DX8cbUBQQ2qTIpK8qrMB1BzGxSYNUw6L?usp=share_link) [:clipboard:](./METAL-tiered_imagenet-Conv64F-5-1.yaml) | 70.40 ± 0.21%           | 71.27 ± 0.33% [:arrow_down:](https://drive.google.com/drive/folders/1d4kXPh6FwcNz0uS_uImTDGPLjEOxDMc-?usp=share_link) [:clipboard:](./METAL-tiered_imagenet-Conv64F-5-5.yaml) |         |
| 2    | ResNet12  | 63.89 ± 0.43%         | 61.77 ± 0.59% [:arrow_down:](https://drive.google.com/drive/folders/1--jVIxv8mlaAPRScKg6UbVvJVAzEn1iJ?usp=share_link) [:clipboard:](./METAL-tiered_imagenet-resnet12-5-1.yaml) | 80.14 ± 0.40%      | 78.58 ± 0.47% [:arrow_down:](https://drive.google.com/drive/folders/1BAZ5NeV1q5N_S0dkOuYJythtILQuaemw?usp=share_link) [:clipboard:](./METAL-tiered_imagenet-resnet12-5-5.yaml) |         |