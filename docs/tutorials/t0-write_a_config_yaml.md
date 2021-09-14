# Write the `.yaml` configuration file

Code for this section:
```
core/config/config.py
config/*
```

## The composition of the LibFewShot configuration file

The LibFewShot configuration file uses a yaml format file and it also supports reading global configuration changes from the command line. We pre-defined a default configuration`core/config/default.yaml`, and the user can put the custom configuration in the `config/` directory，save this file in `yaml` format. At parsing, the sequencing relationship that defines the configuration of the method is `default.yaml->config/->console`. The latter definition overrides the same value in the previous definition.

The most basic configurations are set in `default.yaml`, you can't run a program just by `default.yaml`, before running, the user is required to define the configuration of the methods that have been implemented in LibFewShot in the `config/` directory.

Given that the few-shot menthods have some basic parameters such as `way, shot` or `device id` which are often needed to be changed, but the experience of modifying the `yaml` file is poor, LibFewShot supports making changes to some simple configurations on the command line without modifying the `yaml` file. Similarly, during the training and testing of different methods, many parameters are the same, and for brevity, we wrap these same parameters together and put them in the`config/headers`, so that we can write the `yaml` file of the custom method succinctly by importing them.

The following is the composition of the files in the `config/headers` directory.

- `data.yaml`: The relevant configuration of the data is defined in this file.
- `device.yaml`: The relevant configuration of GPU is defined in this file.
- `losses.yaml`: The relevant configuration of the loss used for training is defined in this file.
- `misc.yaml`: The miscellaneous configuration is defined in this file.
- `model.yaml`: The relevant configuration of the model is defined in this file.
- `optimizer.yaml`: The relevant configuration of the optimizer used for training is defined in this file.

## The settings for the LibFewShot configuration file

The following describes each part of configuration file's information and how to write them, and gives an example of how the DN4 method is configured.

### The settings for data

+ `data_root`: The storage path of the dataset.
+ `image_size`: The size of the data in training will be resize.
+ `use_momery`: whether to use memory to speed up reads.
+ `augment`: whether to use data enhancement.
+ `augment_times：support_set`: The number of times data enhancements were used.Enhancing multiple times is equivalent to expanding the `support set` data.
+ `augment_times_query：query_set`: The number of times data enhancements were used.Enhancing multiple times is equivalent to expanding the `query set` data.

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
  + `kwargs`: The parameters used in the `backbone` initialization, must keep the name consistent with the name in the code.
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

+ `parallel_part`: The parts need to be processed in parallel during forward propagation, and The variable names of the method in these parts are given as a list

+ `pretrain_path`: Pretrain weight addresses that need to be assigned to files. At the beginning of the training, this setting is checked, if not empty, the pre-training weight of the target address is loaded into the `backbone` of the current training.

+ `resume`: If set to True, the training status is read from the default address and continues training.

+ `way_num`: The number of `way` during training.

+ `shot_num`: The number of `shot` during training.

+ `query_num`: The number of `query` during training.

+ `test_way`: The number of `way` during testing. If not specified, the `way_num` is assigned to the `test_way`.

+ `test_shot`: The number of `shot` during testing. If not specified, the `shot_num` is assigned to the `test_way`.

+ `test_query`: The number of `query` during testing. If not specified, the `query_num` is assigned to the `test_way`。

+ `episode_size`: The number of tasks used for network training at a time.

+ `batch_size`: The `batch size` used when the `pre-training` model is `pre-trained`. During training in other kinds of methods, this property is useless.

+ `train_episode`: The number of tasks per `epoch` in training.

+ `test_episode`: The number of tasks per `epoch` in testing.

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

  + `optimizer`: Optimizer information used in training.
  + `name`: The name of tne Optimizer, The framework only temporarily supports all Optimizer provided by `pytorch`.
  + `kwargs`: The parameters of the optimizer need to be passed in, and the name needs to be the same as the parameter name required by the pytorch optimizer.
  + `other`:Currently, the framework only supports the learning rate used by each part of a separately specified method, and the name needs to be the same as the variable name used in the method.

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

+ `lr_scheduler`: the learning rate adjustment strategy used in training, The framework only temporarily supports all the learning rate adjustment strategy provided by `pytorch`.
  + `name`: The name of the learning rate adjustment strategy.
  + `kwargs`: In addition to the optimizer, The name of the learning rate adjustment strategy to pass, same as the parameters are required for the learning rate adjustment strategy in `pytorch`.
  ```yaml
  lr_scheduler:
    name: StepLR
    kwargs:
      gamma: 0.5
      step_size: 10
  ```

  ### The settings for loss

+ `loss`: Loss function information used in training.
  + `name`: The name of tne loss function, The framework only temporarily supports all loss functions provided by `pytorch`.
  + `kwargs`: The parameters required for the loss function, same as the parameters are required for loss functions in `pytorch`.

  ```yaml
  loss:
      name: CrossEntropy
      kwargs: ~
  ```

  ### The settings for Hardware

+ `device_ids`: The `gpu` number, which is the same as the `nvidia-smi` command.
+ `n_gpu`: The number of parallel `gpu` used during training, if `1`, it can't apply to parallel training.
+ `deterministic`: Whether to turn on `torch.backend.cudnn.benchmark` and `torch.backend.cudnn.deterministic` and whether to determine random seeds during training.
+ `seed`: Seed points used in training `numpy`，`torch`，`cuda`.

  ```yaml
  device_ids: 0,1,2,3,4,5,6,7
  n_gpu: 4
  seed: 0
  deterministic: False
  ```

  ### The settings for Miscellaneous

+ `log_name`: if empty, use the auto-generated `classifier.name-data_root-backbone-way_num-shot_num` file directory.
+ `log_level`: The log output level in training.
+ `log_interval`: The number of tasks for the log output interval.
+ `result_root`: The root of the result.
+ `save_interval`: `The epoch interval to save weights.`
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
