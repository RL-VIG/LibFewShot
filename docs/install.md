# 安装
本节给出了从零开始搭建`LibFewShot`可运行的环境的教程。

## 获取LibFewShot

使用以下命令获取LibFewShot:
```shell
cd ~
git clone https://github.com/RL-VIG/LibFewShot.git
```

## 配置`LibFewShot`环境

可以按照下面方法配置环境
1. 创建anaconda环境。
    ```shell
    cd <path-to-LibFewShot> # 进入clone好的LibFewShot目录
    conda create -n libfewshot python=3.7
    conda activate libfewshot
    ```

2. 跟随PyTorch和torchvision的[官方引导](https://pytorch.org/get-started/locally/)进行安装。

3. 依赖包安装
   + pip
    ```shell
    cd <path-to-LibFewShot> # cd 进入`LibFewShot` 目录
    pip install -r requirements.txt
    ```
   + 或者其他安装方式，只要满足：
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

## 测试安装是否正确
1. 修改`run_trainer.py`为
    ```python
    from core.config import Config
    from core.trainer import Trainer

    if __name__ == "__main__":
        config = Config("./config/test_install.yaml").get_config_dict()
        trainer = Trainer(config)
        trainer.train_loop()
    ```
2. 修改`config/headers/data.yaml`中的`data_root`为你的数据集路径
3. 执行
   ```shell
   python run_trainer.py
   ```
4. 若第一个epoch输出正常，则表明`LibFewShot`已成功安装

## 后续
模型训练和代码修改可参考 [训练/测试LibFewShot中已集成的方法](./tutorials/t1-train_and_test_exist_methods.md) 以及其他部分教程。
