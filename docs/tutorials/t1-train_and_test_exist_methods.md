# Train/test existing methods in LibFewShot

Code for this section：
```
config/dn4.yaml
run_trainer.py
run_test.py
```

In this section, we take the DN4 method as an example to describe how to train and test an implemented method.

## Configuration files

In [t0-write_a_config_yaml.md](./t0-write_a_config_yaml.md), we have showed how to write a configuration file. We also assemble some of the common configuration into the public file, so that you can easily finish your DN4 configuration file.

```yaml
includes:
	- headers/data.yaml
	- headers/device.yaml
	- headers/misc.yaml
	- headers/optimizer.yaml
	- backbones/resnet12.yaml
	- classifiers/DN4.yaml
```

For specific customer requirements, you can modify the related included files or use other files and add your own configuration.

## Train

Name the configuration file we have finished in previous section as `dn4.yaml`, place it into the `config/` directory.

Modify the `run_trainer.py` file in project root as follow:

```python
from core.config import Config
from core.trainer import Trainer

if __name__ == "__main__":
    config = Config("./config/dn4.yaml").get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
```

Next, run this instruction in your shell

```shell
python run_trainer.py
```

and the training will start.

## Test

Modify the `run_test.py` file in project root as follow:

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

Input in your shell:

```shell
python run_test.py
```

and the testing will start.

Of course, all of the VAR_DICT variables in`run_test.py` can be removed, by running instruction as follows

```shell
python run_test.py --test_epoch 5 --device_ids 4 --n_gpu 1 --test_episode 600 --episode_size 1
```

to achieve the same effect.
