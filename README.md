# [LibFewShot](https://arxiv.org/abs/2109.04898)
Make few-shot learning easy.

[LibFewShot: A Comprehensive Library for Few-shot Learning](https://arxiv.org/abs/2109.04898).
Wenbin Li, Chuanqi Dong, Pinzhuo Tian, Tiexin Qin, Xuesong Yang, Ziyi Wang, Jing Huo, Yinghuan Shi, Lei Wang, Yang Gao, Jiebo Luo. In arXiv 2021.<br>
<img src='flowchart.png' width=1000/>

## Supported Methods
### Fine-tuning based methods
+ [Baseline (ICLR 2019)](https://arxiv.org/abs/1904.04232)
+ [Baseline++ (ICLR 2019)](https://arxiv.org/abs/1904.04232)
+ [RFS (ECCV 2020)](https://arxiv.org/abs/2003.11539)
+ [SKD (arxiv 2020)](https://arxiv.org/abs/2006.09785)
### Meta-learning based methods
+ [MAML (ICML 2017)](https://arxiv.org/abs/1703.03400)
+ [Versa (NeurIPS 2018)](https://openreview.net/forum?id=HkxStoC5F7)
+ [R2D2 (ICLR 2019)](https://arxiv.org/abs/1805.08136)
+ [LEO (ICLR 2019)](https://arxiv.org/abs/1807.05960)
+ [MTL (CVPR 2019)](https://arxiv.org/abs/1812.02391)
+ [ANIL (ICLR 2020)](https://arxiv.org/abs/1909.09157)
### Metric-learning based methods
+ [ProtoNet (NeurIPS 2017)](https://arxiv.org/abs/1703.05175)
+ [RelationNet (CVPR 2018)](https://arxiv.org/abs/1711.06025)
+ [ConvaMNet (AAAI 2019)](https://ojs.aaai.org//index.php/AAAI/article/view/4885)
+ [DN4 (CVPR 2019)](https://arxiv.org/abs/1903.12290)
+ [CAN (NeurIPS 2019)](https://arxiv.org/abs/1910.07677)
+ [ATL-Net (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/0100.pdf)
+ [ADM (IJCAI 2020)](https://arxiv.org/abs/2002.00153)
+ [FEAT (CVPR 2020)](http://arxiv.org/abs/1812.03664)



## Quick Installation

Please refer to [install.md](https://libfewshot-en.readthedocs.io/en/latest/install.html)([安装](https://libfewshot-en.readthedocs.io/zh_CN/latest/install.html)) for installation.

Complete tutorials can be found at [document](https://libfewshot-en.readthedocs.io/en/latest/)([中文文档](https://libfewshot-en.readthedocs.io/zh_CN/latest/index.html)).

## Datasets
[Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Standford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [Standford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) and [*mini*ImageNet](https://arxiv.org/abs/1606.04080v2) are available at [Google Drive](https://drive.google.com/drive/u/1/folders/1SEoARH5rADckI-_gZSQRkLclrunL-yb0) and [百度网盘(提取码：yr1w)](https://pan.baidu.com/s/1M3jFo2OI5GTOpytxgtO1qA).

## Contributing
Please feel free to contribute any kind of functions or enhancements, where the coding style follows PEP 8. Please kindly refer to [contributing.md](https://libfewshot-en.readthedocs.io/en/latest/contributing.html)([贡献代码](https://libfewshot-en.readthedocs.io/zh_CN/latest/contributing.html)) for the contributing guidelines.

## License
This project is licensed under the MIT License. See LICENSE for more details.

## Acknowledgement
LibFewShot is an open source project designed to help few-shot learning researchers quickly understand the classic methods and code structures. We welcome other contributors to use this framework to implement their own or other impressive methods and add them to LibFewShot. This library can only be used for academic research. We welcome any feedback during using LibFewShot and will try our best to continually improve the library.

## Citation
If you use this code for your research, please cite our paper.
```
@article{li2021LibFewShot,
  title={LibFewShot: A Comprehensive Library for Few-shot Learning},
  author={Li, Wenbin and Dong, Chuanqi and Tian, Pinzhuo and Qin, Tiexin and Yang, Xuesong and Wang, Ziyi and Huo Jing and Shi, Yinghuan and Wang, Lei and Gao, Yang and Luo, Jiebo},
  journal={arXiv preprint arXiv:2109.04898},
  year={2021}
}
```
