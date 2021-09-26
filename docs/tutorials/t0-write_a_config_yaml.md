# Write a `.yaml` configuration file

Code for this section:
```
core/config/config.py
config/*
```

## Composition of the configuration file in LibFewShot

The configuration file of LibFewShot uses a yaml format file and it also supports reading the global configuration changes from the command line. We have pre-defined a default configuration `core/config/default.yaml`. The users can put the custom configuration into the `config/` directory, and save this file in the `yaml` format. At parsing, the sequencing relationship of defining the configuration of the method is `default.yaml->config/->console`. The latter definition overrides the same value in the former definition.

Although most of the basic configurations have been set in the `default.yaml`, you can not directly run a program just using the `default.yaml`. Before running the code, the users are required to define a configuration file of one method that has been implemented in LibFewShot in the `config/` directory.

Considering that FSL menthods usually have some basic parameters, such as `way, shot` or `device id`, which are often needed to be changed, LibFewShot also supports making changes to some simple configurations on the command line without modifying the `yaml` file. Similarly, during training and test, because many parameters are the same of different methods, we wrap these same parameters together and put them into the`config/headers` for brevity. In this way, we can write the `yaml` files of the custom methods succinctly by importing them.

The following is the composition of the files in the `config/headers` directory.

- `data.yaml`: The relevant configuration of the data is defined in this file.
- `device.yaml`: The relevant configuration of GPU is defined in this file.
- `losses.yaml`: The relevant configuration of the loss used for training is defined in this file.
- `misc.yaml`: The miscellaneous configuration is defined in this file.
- `model.yaml`: The relevant configuration of the model is defined in this file.
- `optimizer.yaml`: The relevant configuration of the optimizer used for training is defined in this file.

## The settings of the configuration file in LibFewShot

The following details each part of the configuration file and explain how to write them. An example of how the DN4 method is configured is also presented.

### The settings for data

+ `data_root`: The storage path of the dataset.
+ `image_size`: The size of the input image.
+ `use_momery`: Whether to use memory to accelerate reading.
+ `augment`: Whether to use data augmentation.
+ `augment_times：support_set`: The number of data augmentation/transformations used. Expanding the `support set` data for multiple times.
+ `augment_times_query：query_set`: The number of data augmentation/transformations used. Expanding the `query set` data for multiple times.

  ```yaml
  data_root: /data/miniImageNet--ravi
  image_size: 84
  use_memory: False
  augment: True
  augment_times: 1
  augment_times_query: 1
  ```

### The settings for model

+ `backbone`: The `backbone` information used in the method.
  + `name`: The name of the `backbone`, needs to match the case of the `backbone` implemented in LibFewShot.
  + `kwargs`: The parameters used in the `backbone`, must keep the name consistent with the name in the code.
    + is_flatten: The default is False, and if `True`, the feature vector after flatten is returned.
    + avg_pool: The default is False, and if `True`, the feature vector after `global average pooling` is returned.
    + is_feature: The default is False, and if `True`, the output of each `block` in `backbone` is returned.

  ```yaml
  backbone:
      name: Conv64FLeakyReLU
      kwargs:
          is_flatten: False
  ```

+ `classifier`: The `classifier` information used in the method.
  + `name`: The name of the `classifier`, needs to match the case of the `classifier` implemented in LibFewShot.
  + `kwargs`: The parameters used in the `classifier` initialization, must keep the name consistent with the name in the code.

  ```yaml
  classifier:
      name: DN4
      kwargs:
          n_k: 3
  ```

### The settings for training

+ `epoch`: The number of `epoch` during training.

+ `test_epoch`: The number of `epoch` during testing.

+ `parallel_part`: The parts that need to be processed in parallel in the forward propagation, and the variable names of the method in these parts are given as a list

+ `pretrain_path`: The path of the pre-training weights. At the beginning of the training, this setting will be first checked. If it is not empty, the pre-trained weights of the target path will be loaded into the `backbone` of the current training.

+ `resume`: If set to True, the training status is read from the default address to support continual training.

+ `way_num`: The number of `way` during training.

+ `shot_num`: The number of `shot` during training.

+ `query_num`: The number of `query` during training.

+ `test_way`: The number of `way` during testing. If not specified, the `way_num` is assigned to the `test_way`.

+ `test_shot`: The number of `shot` during testing. If not specified, the `shot_num` is assigned to the `test_way`.

+ `test_query`: The number of `query` during testing. If not specified, the `query_num` is assigned to the `test_way`。

+ `episode_size`: The number of tasks/episodes used for the network training at each time.

+ `batch_size`: The `batch size` used when the `pre-training` model is `pre-trained`. In some kinds of methods, this property is useless.

+ `train_episode`: The number of tasks per `epoch` during training.

+ `test_episode`: The number of tasks per `epoch` during testing.

  ```yaml
  epoch: 50
  test_epoch: 5

  parallel_part:
  - emb_func
  - dn4_layer

  pretrain_path: ~
  resume: False

  way_num: 5
  shot_num: 5
  query_num: 15
  test_way: ~
  test_shot: ~
  test_query: ~
  episode_size: 1
  # batch_size only works in pre-train
  batch_size: 128
  train_episode: 10000
  test_episode: 1000
  ```

  ### The settings for optimizer

  + `optimizer`: Optimizer information used during training.
  + `name`: The name of the Optimizer, only temporarily supports all Optimizers provided by `PyTorch`.
  + `kwargs`: The parameters used in the optimizer, and the name needs to be the same as the parameter name required by the pytorch optimizer.
  + `other`: Currently, the framework only supports the learning rate used by each part of a separately specified method, and the name needs to be the same as the variable name used in the method.

  ```yaml
  optimizer:
      name: Adam
      kwargs:
          lr: 0.01
      other:
          emb_func: 0.01
          #For demonstration purposes, there are no additional training parameters for dn4.
          dnf_layer: 0.001
  ```

+ `lr_scheduler`: The learning rate adjustment strategy used during training, only temporarily supports all the learning rate adjustment strategies provided by `PyTorch`.
  + `name`: The name of the learning rate adjustment strategy.
  + `kwargs`: Other parameters used in the learning rate adjustment strategy in `PyTorch`.
  
  ```yaml
  lr_scheduler:
    name: StepLR
    kwargs:
      gamma: 0.5
      step_size: 10
  ```

  ### The settings for Hardware

+ `device_ids`: The `gpu` number, which is the same as the `nvidia-smi` command.
+ `n_gpu`: The number of parallel `gpu` used during training, if `1`, it can't apply to parallel training.
+ `deterministic`: Whether to turn on `torch.backend.cudnn.benchmark` and `torch.backend.cudnn.deterministic` and whether to determine random seeds during training.
+ `seed`: Seed points used in `numpy`，`torch`，and `cuda`.

  ```yaml
  device_ids: 0,1,2,3,4,5,6,7
  n_gpu: 4
  seed: 0
  deterministic: False
  ```

  ### The settings for Miscellaneous

+ `log_name`: If empty, use the auto-generated `classifier.name-data_root-backbone-way_num-shot_num` file directory.
+ `log_level`: The log output level during training.
+ `log_interval`: The number of tasks for the log output interval.
+ `result_root`: The root of the result.
+ `save_interval`: The epoch interval to save weights.
+ `save_part`: The name of the variable in the method that needs to be saved. Variables with these names are saved separately when the model is saved. The parts that need to be saved are given as a list under `save_part`.

  ```yaml
  log_name: ~
  log_level: info
  log_interval: 100
  result_root: ./results
  save_interval: 10
  save_part:
      - emb_func
      - dn4_layer
  ```
