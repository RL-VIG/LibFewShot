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
2. 在该文件中写入以下指令
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
1. 修改`run_trainer.py`中`config`配置语句为
    ```python
    config = Config("./config/getting_started.yaml").get_config_dict()
    ```
2. 执行
   ```shell
   python run_trainer.py
   ```
3. 等待程序运行结束（你可以去喝100杯咖啡）

## 查看运行日志文件
程序运行完毕之后，可以找到链接`results/ProtoNet-miniImageNet-Conv64F-5-1`和目录`results/ProtoNet-miniImageNet-Conv64F-5-1-$TS`，其中`TS`表示时间戳。目录包含两个文件夹`checkpoint/`和`log_files/`和一个配置文件`config.yaml`。当你多次训练同一种小样本学习方法，链接总会关联到最后创建的目录。

`config.yaml`即本次训练使用的配置文件内容。

`log_files`包含tensorboard记录文件，以及在本模型上的训练日志以及测试日志。

`checkpoints`包含按照save_interval保存的模型文件、最后模型文件（用于resume）和最佳模型文件（用于测试）。模型文件一般分为`emb_func.pth`,`classifier.pth`以及`model.pth`（前两者的组合）
