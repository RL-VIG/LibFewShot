# Self-supervised Knowledge Distillation for Few-shot Learning
## Introduction
| Name:    | [SKD](https://arxiv.org/abs/2006.09785)                          |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ResNet18 |
| Type:    | Fine-tuning       |
| Venue:   | arXiv'20                      |
| Codes:   | [**SKD**](https://github.com/brjathu/SKD)|

+ Due to lacking enough GPUs, considering SKD-Gen0 uses episodic tasks for val/test only, we believe that the best model chosen by 5-1 validation is almost the best model chosen by 5-5, so we run SKD-Gen0 once for 2 testing setting.

Cite this work with:
```bibtex
@article{DBLP:journals/corr/abs-2006-09785,
  author    = {Jathushan Rajasegaran and
               Salman Khan and
               Munawar Hayat and
               Fahad Shahbaz Khan and
               Mubarak Shah},
  title     = {Self-supervised Knowledge Distillation for Few-shot Learning},
  journal   = {CoRR},
  volume    = {abs/2006.09785},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.09785},
  archivePrefix = {arXiv},
  eprint    = {2006.09785}
}

```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 48.14 ± 0.33 [:arrow_down:](https://drive.google.com/drive/folders/17NQoyMUTgMNG6TpLUJGvz-mgCi1mViaP?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-Conv64F-Gen0-Table2.yaml) | - | 66.36 ± 0.29 [:arrow_down:](https://drive.google.com/drive/folders/17NQoyMUTgMNG6TpLUJGvz-mgCi1mViaP?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-Conv64F-Gen0-Table2.yaml) | SKD-Gen0-Table2 |
| 2 | ResNet12 | - | 66.40 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/1Iu0w0gTCDgqC48H4-osmZeqkHqHdp1eI?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-resnet12-Gen0-Table2.yaml) | - | 83.06 ± 0.24[:arrow_down:](https://drive.google.com/drive/folders/1Iu0w0gTCDgqC48H4-osmZeqkHqHdp1eI?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-resnet12-Gen0-Table2.yaml) | SKD-Gen0-Table2 |
| 3 | ResNet12 | - | 67.35 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1Uvfcb8CdrkJUKztg4oVL-bY7XifXTPNl?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-resnet12-Gen1-Table2.yaml) | - | 83.31 ± 0.24 [:arrow_down:](https://drive.google.com/drive/folders/1Uvfcb8CdrkJUKztg4oVL-bY7XifXTPNl?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-resnet12-Gen1-Table2.yaml) | SKD-Gen1-Table2 |
| 4 | ResNet18 | - | 66.18 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1QCG9Dr4BPfmEzWUgHe4VqhohfeyR4Jy1?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-resnet18-Gen0-Table2.yaml) | - | 82.21 ±  0.24[:arrow_down:](https://drive.google.com/drive/folders/1QCG9Dr4BPfmEzWUgHe4VqhohfeyR4Jy1?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-resnet18-Gen0-Table2.yaml) | SKD-Gen0-Table2 |
| 5 | ResNet18 | - | 66.70 ± 0.37 [:arrow_down:](https://drive.google.com/drive/folders/1PtKM7hx7rBIjYpn2-4XMDhIsZq4OBjzn?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-resnet18-Gen1-Table2.yaml) | - | 82.60 ±  0.24[:arrow_down:](https://drive.google.com/drive/folders/1PtKM7hx7rBIjYpn2-4XMDhIsZq4OBjzn?usp=sharing) [:clipboard:](./SKDModel-miniImageNet--ravi-resnet18-Gen1-Table2.yaml) | SKD-Gen1-Table2 |


|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | - | 51.78 ± 0.36 [:arrow_down:](https://drive.google.com/drive/folders/1pJQpE53HhL2P4DEleqLDXK-ZtjbKhTpI?usp=sharing) [:clipboard:](./SKDModel-tiered_imagenet-Conv64F-Gen0-Table2.yaml) | - | 70.65 ± 0.32 [:arrow_down:](https://drive.google.com/drive/folders/1pJQpE53HhL2P4DEleqLDXK-ZtjbKhTpI?usp=sharing) [:clipboard:](./SKDModel-tiered_imagenet-Conv64F-Gen0-Table2.yaml) | SKD-Gen0-Table2 |
| 2 | ResNet12 | - | 71.90 ± 0.39 [:arrow_down:](https://drive.google.com/drive/folders/14K4e2m9VzkmTmIldBdOgu59OBJP0MGs2?usp=sharing) [:clipboard:](./SKDModel-tiered_imagenet-resnet12-Table2.yaml) | - | 86.20 ± 0.27 [:arrow_down:](https://drive.google.com/drive/folders/14K4e2m9VzkmTmIldBdOgu59OBJP0MGs2?usp=sharing) [:clipboard:](./SKDModel-tiered_imagenet-resnet12-Table2.yaml) | SKD-Gen0-Table2 |
| 3 | ResNet18 | - | 70.00 ± 0.57 [:arrow_down:](https://drive.google.com/drive/folders/1m4gTAHp6G8UoneJccymGmbuAHvT3pMBv?usp=sharing) [:clipboard:](./SKDModel-tiered_imagenet-resnet18-Gen0-Table2.yaml) | - | 84.70 ± 0.41 [:arrow_down:](https://drive.google.com/drive/folders/1m4gTAHp6G8UoneJccymGmbuAHvT3pMBv?usp=sharing) [:clipboard:](./SKDModel-tiered_imagenet-resnet18-Gen0-Table2.yaml) | SKD-Gen0-Table2 |
