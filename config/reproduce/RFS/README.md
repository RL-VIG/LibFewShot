# Rethinking Few-Shot Image Classification: A Good Embedding is All You Need?
## Introduction
| Name:    | [RFS](https://arxiv.org/abs/2003.11539)                          |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ResNet18 |
| Type:    | Fine-tuning       |
| Venue:   | ECCV'20                      |
| Codes:   | [**rfs**](https://github.com/WangYueFt/rfs)|

+ When testing the `RFS-simple`, you need to change `test-shot` to `1 or 5` for different setting with 1 checkpoint.

Cite this work with:
```bibtex
@inproceedings{DBLP:conf/eccv/TianWKTI20,
  author    = {Yonglong Tian and
               Yue Wang and
               Dilip Krishnan and
               Joshua B. Tenenbaum and
               Phillip Isola}
  title     = {Rethinking Few-Shot Image Classification: {A} Good Embedding is All
               You Need?},
  booktitle = {Computer Vision - {ECCV} 2020 - 16th European Conference, Glasgow,
               UK, August 23-28, 2020, Proceedings, Part {XIV}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12359},
  pages     = {266--282},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-58568-6_16},
  doi       = {10.1007/978-3-030-58568-6_16}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12  [^metaopt] | 62.02 ± 0.63 | 62.80 ± 0.52 [:arrow_down:](https://drive.google.com/drive/folders/1COMUhto08xtSaOazlMw1GndEAfeIZ-XQ?usp=sharing)  [:clipboard:](./RFS-simple-miniImageNet-ResNet12M-Table1.yaml) | 79.64 ± 0.44 | 79.57± 0.39 [:arrow_down:](https://drive.google.com/drive/folders/1COMUhto08xtSaOazlMw1GndEAfeIZ-XQ?usp=sharing) [:clipboard:](./RFS-simple-miniImageNet-ResNet12M-Table1.yaml) | rfs-simple-Table-1 |

[^metaopt]: ResNet12-MetaOpt with [64,160,320,640].
