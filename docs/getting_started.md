# 入门

本节展示了使用`LibFewShot`的流程示例。

## 准备数据集（以miniImagenet为例）
1. 下载并解压[miniimagent--ravi](TODO)
2. 检查数据集格式：
    数据集应该具有以下格式(其它数据集如tieredImageNet等也是)
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


## 修改配置文件
以`ProtoNet`为例：
1. 在`config`目录下新建一个yaml文件`getting_started.yaml`
2. 在该文件中写入以下内容
   ```yaml
   includes:
     - headers/data.yaml
     - headers/device.yaml
     - headers/losses.yaml
     - headers/misc.yaml
     - headers/model.yaml
     - headers/optimizer.yaml
     - classifiers/Proto.yaml
     - backbones/Conv64FLeakyReLU.yaml
      ```

更细节的部分可参考 [编写`.yaml`配置文件](./tutorials/t0-write_a_config_yaml.md)。

## 运行
1. 修改`run_trainer.py`为
    ```python
    from core.config import Config
    from core.trainer import Trainer

    if __name__ == "__main__":
        config = Config("./config/getting_started.yaml").get_config_dict()
        trainer = Trainer(config)
        trainer.train_loop()
    ```
2. 执行
   ```shell
   python run_trainer.py
   ```
3. 等待程序运行结束（你可以去喝100杯咖啡）

## 查看运行日志文件
程序运行完毕之后，可以在`results`目录下找到使用时间戳作为标记的对应文件夹，如`TODOTODO`，其中包含两个文件夹`checkpoints`和`log_files`以及一个配置文件`config.yaml`。

`config.yaml`即本次训练使用的配置文件内容。

`log_files`包含tensorboard记录文件，以及在本模型上的训练日志以及测试日志。

`checkpoints`包含按照save_interval保存的模型文件、最后模型文件（用于resume）和最佳模型文件（用于测试），模型文件一般分为`model_func.pth`,`classifier.pth`以及这两者的组合。
