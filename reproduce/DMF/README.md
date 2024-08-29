# Learning Dynamic Alignment via Meta-filter for Few-shot Learning
## Introduction
| Name:    | [DMF](https://arxiv.org/pdf/2103.13582) |
|----------|-------------------------------|
| Embed.:  | Conv64F/ResNet12/ |
| Type:    | Metric       |
| Venue:   | CVPR'21                      |
| Codes:   | [**DMF**](https://github.com/chmxu/Dynamic-Meta-filter) |


Cite this work with:
```bibtex
@inproceedings{xu2021dmf,
  title={Learning Dynamic Alignment via Meta-filter for Few-shot Learning},
  author={Chengming Xu and Chen Liu and Li Zhang and Chengjie Wang and Jilin Li and Feiyue Huang and Xiangyang Xue and Yanwei Fu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
---
## Setup
You need to setup first. Run the code below:
```
python DMF_setup.py develop build
```
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12 | 67.76 ± 0.46% | 67.185% [:arrow_down:]( https://pan.baidu.com/s/1H_Y7G4BH-OnbU5hl54ljww?pwd=s3ut) [:clipboard:](./META_FILTER-miniImagenet-resnet12_drop-5-1.yaml) | 82.71 ± 0.31% | 81.997% [:arrow_down:]( https://pan.baidu.com/s/1X99febUlbV7WE6IYkNeGfQ?pwd=8rv5) [:clipboard:](./META_FILTER-miniImagenet-resnet12_drop-5-5.yaml) | Comments |

|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12 | 71.89 ± 0.52% | 71.369% [:arrow_down:]( https://pan.baidu.com/s/1pYD9H7SOuw0BYIQerYjOhA?pwd=546y) [:clipboard:](./META_FILTER-tiered_imagenet-resnet12_drop-5-1.yaml) | 85.96 ± 0.35% | 85.350% [:arrow_down:](https://pan.baidu.com/s/1Jf1XKcXziEdcXGZjFwQdeg?pwd=hy7j) [:clipboard:](./META_FILTER-tiered_imagenet-resnet12_drop-5-5.yaml) | Comments |