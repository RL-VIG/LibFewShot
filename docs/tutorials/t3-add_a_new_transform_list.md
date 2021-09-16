# Transformations

Code for this section：
```
core/data/dataloader.py
core/data/collates/contrib/__init__.py
core/data/collates/collate_functions.py
```

In `LibFewShot`，we use a base `transforms` to compare some methods fairly. The base `transforms` could be divided into 3 sub-transforms:

```
Resize&Crop + ExtraTransforms + ToTensor&Norm
```
There are some difference in`Resize&Crop` for different dataset and config file (key `augment`):

1. during train phase and `config.augment` is `True`：
   ```python
   from torchvision import transforms
   transforms.RandomResizedCrop((config.image_size, config.image_size))
   ```
2. other phase：
   ```python
   from torchvision import transforms
   transforms.Resize((96, 96))  # or 256 when config.image_size = 224
   transforms.CenterCrop((84, 84)) # or 224 when config.image_size = 224
   ```

Besides, you may notice that `ToTensor & Norm` always uses the same set of mean and variance, then you can reset mean and variance for different datasets.

```python
MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
```
