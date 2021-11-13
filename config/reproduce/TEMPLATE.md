# Template for Reproduce configs
## Introduction
| Name:    | [LibFewShot](Link-to-paper) (template)         |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ResNet18/WRN |
| Type:    | Metric/Meta/Fine-tuning       |
| Venue:   | arXiv'21                      |
| Codes:   | [**Bold if official**](https://github.com/RL-VIG/LibFewShot), [*Repo-Name(Italic if unofficial)*](link)                   |
*Tutorials for train and test this method (if needed)*

Cite this work with (template):
```bibtex
@article{li2021LibFewShot,
  title={LibFewShot: A Comprehensive Library for Few-shot Learning},
  author={Li, Wenbin and Dong, Chuanqi and Tian, Pinzhuo and Qin, Tiexin and Yang, Xuesong and Wang, Ziyi and Huo Jing and Shi, Yinghuan and Wang, Lei and Gao, Yang and Luo, Jiebo},
  journal={arXiv preprint arXiv:2109.04898},
  year={2021}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | Conv64F | 50.00 ± 0.05 | 50.00 ± 0.05 [:arrow_down:](Link-to-model-url) [:clipboard:](Link-to-config-url) | 50.00 ± 0.05 | 50.00 ± 0.05 | Comments |
| 2 | ResNet12[^1] | 50.00 ± 0.05 | 50.00 ± 0.05 | 50.00 ± 0.05 | 50.00 ± 0.05 | Comments |
| 3 | ResNet12[^metaopt] | 50.00 ± 0.05 | 50.00 ± 0.05 | 50.00 ± 0.05 | 50.00 ± 0.05 | Comments |




**CrossDomain**

|   | Embedding | *mini* -> CUB (5,1) | *mini* -> CUB (5,5) |
|---|-----------|--------------------|--------------------|
| 1 | Conv64F   | 50.00 ± 0.05       | 50.00 ± 0.05       |
| 2 | ResNet12[^1]| 50.00 ± 0.05       | 50.00 ± 0.05       |
| 3 | ResNet12[^metaopt] | 50.00 ± 0.05       | 50.00 ± 0.05       |

[^1]: (Ordered footnote) ResNet12-TADAM with [64,128,256,512]
[^metaopt]: (Named footnote) ResNet12-MetaOpt with [64,160,320,640]
