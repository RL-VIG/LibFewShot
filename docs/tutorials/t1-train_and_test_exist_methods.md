# Train and Test an exist method

本部分以DN4方法为例，介绍如何训练和测试一个已经实现好的方法。

## 配置文件

从`t0`中我们介绍了如何编写配置文件。并且我们将一部分的常用配置集合成了一个文件，因此可以简单地完成`DN4`配置文件的编写。

```yaml
includes:
	- headers/data.yaml
	- headers/device.yaml
	- headers/misc.yaml
	- headers/optimizer.yaml
	- backbones/resnet12.yaml
	- classifiers/dn4_classifier.yaml
```

如果有自定义需要，也可以修改对应的`includes`下的导入文件中的内容。也可以删除对应的`includes`下的导入文件，自行添加对应的值。

## 训练

将上一步编写的配置文件命名为`dn4.yaml`，放到`config/`目录下。

修改根目录下的`run_trainer.py`文件。

```python
from core.config import Config
from core.trainer import Trainer

if __name__ == "__main__":
    config = Config("./config/dn4.yaml").get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
```

接着，在shell中输入

```shell
python run_trainer.py
```

即可开启训练过程。

## 测试

修改根目录下的`run_test.py`文件。

```python
import os
from core.config import Config
from core.test import Test

PATH = "./results/DN4-miniImageNet-resnet12-5-5"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "4",
    "n_gpu": 1,
    "test_episode": 600,
    "episode_size": 1,
}

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    test = Test(config, PATH)
    test.test_loop()

```

在shell中运行

```shell
python run_test.py
```

即可开始测试过程。

当然，上述run_test.py中的VAR_DICT变量中的值都可以去掉，然后通过在shell中运行

```shell
python run_test.py --test_epoch 5 --device_ids 4 --n_gpu 1 --test_episode 600 --episode_size 1
```

来达到同样的效果。
