# MatchingNet with IFSL for Few-Shot learning
## Introduction
| Name:    | [MatchingNet](https://arxiv.org/abs/1606.04080)                          |
|----------|-------------------------------|
| Embed.:  | ResNet12 |
| Type:    | Meta       |
| Venue:   | NeurIPS'16                      |
| Codes:   | [**MatchingNetworks**](https://github.com/gitabcworld/MatchingNetworks)|

+ In addition to using MatchingNet's meta-learning method, we also added [another paper's](https://proceedings.neurips.cc/paper/2020/hash/1cc8a8ea51cd0adddf5dab504a285915-Abstract.html) IFSL method to modify backbone knowledge in the process of transferring to fine-tuning

Cite this work with:
```bibtex
@article{vinyals2016matching,
  title={Matching networks for one shot learning},
  author={Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and Wierstra, Daan and others},
  journal={Advances in neural information processing systems},
  volume={29},
  year={2016}
}
@article{yue2020interventional,
  title={Interventional few-shot learning},
  author={Yue, Zhongqi and Zhang, Hanwang and Sun, Qianru and Hua, Xian-Sheng},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={2734--2746},
  year={2020}
}
```
---
## Results and Models

**Classification**

|   | Embedding | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1) | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12 | - | 61.022 ± 0.35 [:arrow_down:](https://pan.baidu.com/s/1Ze56sk_3pCI4v7sq6HL2PA) code: 8yd8 | - |  75.051 ± 0.30 [:arrow_down:](https://pan.baidu.com/s/1WoCT34hWURGOKbRJ4CFjUQ ) code: 28lr | Table2 |

|   | Embedding | :book: *tiered*ImageNet (5,1) | :computer: *tiered*ImageNet (5,1) | :book:*tiered*ImageNet (5,5) | :computer: *tiered*ImageNet (5,5) | :memo: Comments  |
|---|-----------|--------------------|--------------------|--------------------|--------------------|---|
| 1 | ResNet12 | - | 67.56 ± 0.19 [:arrow_down:](https://pan.baidu.com/s/1oSdllUS-Juo-f77WWtzXig) code: gofr | - | 81.85 ± 0.21 [:arrow_down:](https://pan.baidu.com/s/1a8VSRX2XdVojTi3H7ulQ1g ) code: kx9m | Table.2 |
