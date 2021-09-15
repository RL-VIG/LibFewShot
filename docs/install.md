## Installation

This section provides a tutorial on building a working environment for `LibFewShot` from scratch.

## Get `LibFewShot` library

Use the following command to get `LibFewShot`:

```shell
cd ~
git clone https://github.com/RL-VIG/LibFewShot.git
```

## Configure the `LibFewShot` environment

The environment can be configured in any of the following ways:

1. conda(recommend)
    ```shell
    cd <path-to-LibFewShot> # cd in `LibFewShot` directory
    conda env create -f requirements.yaml
    ```

2. pip
    ```shell
    cd <path-to-LibFewShot> # cd in `LibFewShot` directory
    pip install -r requirements.txt
    ```
3. or whatever works for you as long as the following package version conditions are meet:
    ```
    numpy >= 1.19.5
    pandas >= 1.1.5
    Pillow >= 8.1.2
    PyYAML >= 5.4.1
    scikit-learn >= 0.24.1
    scipy >= 1.5.4
    tensorboard >= 2.4.1
    torch >= 1.5.0
    torchvision >= 0.6.0
    python >= 3.6.0
    ```

## Test the installation

1. modify `run_trainer.py:10`
    ```python
    # -*- coding: utf-8 -*-
    import sys

    sys.dont_write_bytecode = True

    from core.config import Config
    from core import Trainer

    if __name__ == "__main__":
        config = Config("./config/test_install.yaml").get_config_dict()
        trainer = Trainer(config)
        trainer.train_loop()
    ```
2. modify `data_root` in `config/headers/data.yaml` to the path of the dataset to be used.
3. run code
   ```shell
   python run_trainer.py
   ```
4. If the first output is correct then `LibFewShot` has been installed successfully.

## Next

For model training and code modification, please see the [train/test methods already integrated in LibFewShot](./tutorials/t1-train_and_test_exist_methods.md) and other sections of the tutorial.
