## Installation

This section provides a tutorial on building a working environment for `LibFewShot` from scratch.

## Get the `LibFewShot` library

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


1. set the `config` as follows in `run_trainer.py`:
    ```python
    config = Config("./config/test_install.yaml").get_config_dict()
    ```
2. modify `data_root` in `config/headers/data.yaml` to the path of the dataset to be used.
3. run code
   ```shell
   python run_trainer.py
   ```
4. If the first output is correct, it means that `LibFewShot` has been successfully installed.

## Next

For model training and code modification, please see the [train/test methods already integrated in LibFewShot](./tutorials/t1-train_and_test_exist_methods.md) and other sections of the tutorial.
