# Use Dataset

In `LibFewShot`, datasets have a fixed representation format. We read the data according to the datasets in most few-shot learning settings, like [*mini*ImageNet](https://paperswithcode.com/dataset/miniimagenet-1) and [*tiered*ImageNet](https://paperswithcode.com/dataset/tieredimagenet). Some datasets like
[Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) can be downloaded from the Internet and unzipped for using directly.

If you want to use a new dataset but its data format is different from the above datasets, you need to transform it into the same dataset format.

## Dataset format
Like *mini*ImageNet, dataset format should be the same as follow:
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

All training, evaluating and testing images should be placed in the `images` directory, respectively using `train.csv`，`test.csv` and `val.csv` files to split the dataset. These three files have similar format, required for data organization as follow:

```csv
filename    , label
images_m.jpg, class_name_i
...
images_n.jpg, class_name_j
```
The CSV head contains only two columns, each for filename and label. The filename should be relative

path from `images` directory. Which means, for an image with absolute path `.../dataset_folder/images/images_1.jpg`，its `filename` should be `images_1.jpg`. In a similar way, for an image with absolute path  `.../dataset_folder/images/class_name_1/images_1.jpg`,  its `filename` should be `class_name_1/images_1.jpg`.

## Configure Dataset
After downloading dataset and transforming it into above dataset format, you only need to change the `data_root` in the configuration file. Notice that `LibeFewShot` will  print the data directory's name as dataset name in log.
