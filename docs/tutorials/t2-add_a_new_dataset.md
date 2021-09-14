# 使用数据集

在`LibFewShot`中，数据集有固定的表示格式。我们按照大多数小样本学习设置下的数据集格式进行数据的读取，例如 [*mini*ImageNet](https://paperswithcode.com/dataset/miniimagenet-1) 和 [*tiered*ImageNet](https://paperswithcode.com/dataset/tieredimagenet) ，因此例如
[Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html)
等数据集只需从网络上下载并解压就可以使用。如果你想要使用一个新的数据集，并且该数据集的数据形式与以上数据集不同，那么你需要自己动手将其转换成相同的格式。

## 数据集格式
与 *mini*ImageNet 一样，数据集的格式应该和下面的示例一样：
```
dataset_folder/
├── images/
│   ├── images_1.jpg
│   ├── ...
│   └── images_n.jpg
├── train.csv *
├── test.csv *
└── val.csv *
```

所有的训练、验证以及测试图像都需要放置在`images`文件夹下，通过`train.csv`，`test.csv`和`val.csv`文件分割数据集。三个文件的格式都类似，需要以下面的格式进行数据的组织：
```csv
filename    , label
images_m.jpg, class_name_i
...
images_n.jpg, class_name_j
```
需要保留CSV文件的表头，仅分为文件名和类名两列。这里文件名的路径只写`images`文件夹下的路径，即如果存在这样的文件：`.../dataset_folder/images/images_1.jpg`，那么`filename`字段就需要填写`images_1.jpg`，同理，如果存在这样的文件：`.../dataset_folder/images/class_name_1/images_1.jpg`，那么`filename`字段就需要填写`class_name_1/images_1.jpg`

## 配置数据集
当下载好或按照上述格式整理好数据集后，只需要在配置文件中修改`data_root`字段即可，注意`LibeFewShot`会将数据集文件夹名当作数据集名称打印在log上。
