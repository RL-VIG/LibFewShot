# Cross Attention Network for Few-shot Classification
## Introduction
| Name:    | [CAN](https://arxiv.org/abs/1910.07677)  |
|----------|-------------------------------|
| Embed.:  | Conv64F |
| Type:    | Metric       |
| Venue:   | NeurIPS'17                      |
| Codes:   | [**fewshot-CAN**](https://github.com/blue-blue272/fewshot-CAN) |

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/nips/HouCMSC19,
  author    = {Ruibing Hou and
               Hong Chang and
               Bingpeng Ma and
               Shiguang Shan and
               Xilin Chen},
  title     = {Cross Attention Network for Few-shot Classification},
  booktitle = {Advances in Neural Information Processing Systems 32: Annual Conference
               on Neural Information Processing Systems 2019, NeurIPS 2019, December
               8-14, 2019, Vancouver, BC, Canada},
  pages     = {4005--4016},
  year      = {2019},
  url       = {https://proceedings.neurips.cc/paper/2019/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 55.88 ± 0.38[:arrow_down:](https://drive.google.com/drive/folders/1witI_r60wLVoeoexmpYcmowencqXbUqx?usp=sharing) [:clipboard:](./CAN-miniImageNet--ravi-Conv64F-5-1-Table2.yaml) | - | 70.98 ± 0.30[:arrow_down:](https://drive.google.com/drive/folders/1WqQOqoLtWdXqCxaA1Eek5ZLPemsm5IIM?usp=sharing) [:clipboard:](./CAN-miniImageNet--ravi-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 59.82 ± 0.38 [:arrow_down:](https://drive.google.com/drive/folders/1N7TPFrbJTT8Hk1npUwHyGlVzrNqKO0qf?usp=sharing) [:clipboard:](./CAN-miniImageNet--ravi-resnet12-5-1-Table2.yaml) | - | 76.54 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1IbF1a2HVbJs6uLvSwpX7m6X8XH1wJSc8?usp=sharing) [:clipboard:](./CAN-miniImageNet--ravi-resnet12-5-5-Table2.yaml) | Table.2 |
| 3 | ResNet18 | - | 60.78 ± 0.40 [:arrow_down:](https://drive.google.com/drive/folders/1HURosgYDniFbTOdl02Z9ZvrgpsxBEz1s?usp=sharing) [:clipboard:](./CAN-miniImageNet--ravi-resnet18-5-1-Table2.yaml) | - | 75.05 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/1ydlya4qa_mNIcNfftogqfxEFfemfFUoS?usp=sharing) [:clipboard:](./CAN-miniImageNet--ravi-resnet18-5-5-Table2.yaml) | Table.2 (HW=11)|


|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 55.96 ± 0.42 [:arrow_down:](https://drive.google.com/drive/folders/1zr1s6f1CnfXVbumQ2xmOHTHTKZZugADT?usp=sharing) [:clipboard:](./CAN-tiered_imagenet-Conv64F-5-1-Table2.yaml) | - | 70.52 ± 0.35 [:arrow_down:](https://drive.google.com/drive/folders/1wFzz0RkdTdFN9-mIIuAilKhPslX5gutM?usp=sharing) [:clipboard:](./CAN-tiered_imagenet-Conv64F-5-5-Table2.yaml) | Table.2 |
| 2 | ResNet12 | - | 70.46 ± 0.43 [:arrow_down:](https://drive.google.com/drive/folders/1GLQNF_uodtupwZm-EhhITF8Q9zoGB1uJ?usp=sharing) [:clipboard:](./CAN-tiered_imagenet-resnet12-5-1-Table2.yaml) | - | 84.50 ± 0.30 [:arrow_down:](https://drive.google.com/drive/folders/1YZQLeTQCoNtW_NxX1AINc-LIQt37UqGU?usp=sharing) [:clipboard:](./CAN-tiered_imagenet-resnet12-5-5-Table2.yaml) | Table.2 |
| 3 | ResNet18 | - | 71.70 ± 0.43 [:arrow_down:](https://drive.google.com/drive/folders/1mUoTy-VE1xZJXrrf2BUwwl00XB6A8RVn?usp=sharing) [:clipboard:](./CAN-tiered_imagenet-resnet18-5-1-Table2.yaml) | - | 84.61 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1tvz-ud2GL1c_RuIbwxEV4hp8zIokYJRA?usp=sharing) [:clipboard:](./CAN-tiered_imagenet-resnet18-5-5-Table2.yaml) | Table.2 (HW=11)|
