# Template for Reproduce configs
## Introduction
| Name:    | [MeTal](https://arxiv.org/abs/2110.03909)                                                               |
|----------|---------------------------------------------------------------------------------------------------------|
| Embed.:  | Conv64F,ResNet12                                                                                        |
| Type:    | Meta                                                                                                    |
| Venue:   | arXiv'21                                                                                                |
| Codes:   | [**MeTal**](https://github.com/baiksung/MeTAL)|
Cite this work with:
```bibtex
@InProceedings{baik2021meta,
 title={Meta-Learning with Task-Adaptive Loss Function for Few-Shot Learning},
 author={Sungyong Baik, Janghoon Choi, Heewon Kim, Dohee Cho, Jaesik Min, Kyoung Mu Lee}
 booktitle = {International Conference on Computer Vision (ICCV)}, 
 year={2021}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1)                                                                                                                                                 | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5)                                                                                      | :memo: Comments |
|---|----------|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|----------------------------------------------------------------------------------------------------------------------|-----------------|
| 1 | Conv64F | -                           | 52.364 [:arrow_down:](https://drive.google.com/file/d/1ljtq5PH7VywDh2ZInqWzCOn5Lowu0zyC/view?usp=drive_link) [:clipboard:](./METAL-miniImageNet--ravi-Conv64F-5-1-Table2.yaml)  | -                          | 70.421  [:arrow_down:](https://drive.google.com/file/d/1lzgeg4ckxSP1Zu-E_f4gfMenkK49_2tV/view?usp=drive_link) [:clipboard:](./METAL-miniImageNet--ravi-Conv64F-5-5-Table2.yaml) | Table.2         |
| 2 | ResNet12 | -                           | 60.542 [:arrow_down:](https://drive.google.com/file/d/1qLrWig2eq85wxXkZrP6XGzKqnL6RO3IS/view?usp=drive_link) [:clipboard:](./METAL-miniImageNet--ravi-resnet12-5-1-Table2.yaml) | -                          | 76.880 [:arrow_down:](https://drive.google.com/file/d/1fNUAd9gpKHUeoOSkkVzQj9BPmITnFQEx/view?usp=drive_link) [:clipboard:](./METAL-miniImageNet--ravi-resnet12-5-5-Table2.yaml)         | Table.2        |

