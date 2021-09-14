# Transformations

本节相关代码：
```
core/data/dataloader.py
core/data/collates/contrib/__init__.py
core/data/collates/collate_functions.py
```

在LFS中，我们使用一个基础Transform的结构，以公平的比较多种方法。该基础的Transform结构可分为三段：
```
Resize&Crop + ExtraTransforms + ToTensor&Norm
```
`Resize&Crop`部分根据不同的数据集和配置文件设置(`augment`字段)存在一些差异：
1. 当数据集为训练数据集（train）且`config.augment = True`的时候，使用：
   ```python
   from torchvision import transforms
   transforms.Resize((96, 96)) # 当 config.image_size 为224时，该项为256
   transforms.RandomCrop((84, 84)) # 当 config.image_size 为224时，该项为224
   ```
2. 其他情况下使用：
   ```python
   from torchvision import transforms
   transforms.Resize((96, 96)) # 当 config.image_size 为224时，该项为256
   transforms.CenterCrop((84, 84)) # 当 config.image_size 为224时，该项为224
   ```
另外，你可能注意到在


`ToTensor & Norm`部分使用同一组均值和方差，你可以根据数据集特性重新设置该值：
```python
MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
```
