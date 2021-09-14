# Add a new classifier

Code for this section：
```
core/model/abstract_model.py
core/model/meta/*
core/model/metric/*
core/model/pretrain/*
```

We need to select one representative method from `matric based` methods, `meta learning` methods and `fine-tuning` methods respectively, and describe how to add new methods of the three categories.

Before this，we need to introduce a parent class of all methods: `abstract_model`.

```python
class AbstractModel(nn.Module):
    def __init__(self,...)
    	# base info

    @abstractmethod
    def set_forward(self,):
        # inference phase
        pass

    @abstractmethod
    def set_forward_loss(self,):
        # training phase
        pass

    def forward(self, x):
        out = self.emb_func(x)
        return out

    def train(self,):
        # override super's function

    def eval(self,):
        # override super's function

    def _init_network(self,):
        # init all layers

    def _generate_local_targets(self,):
        # formate the few shot labels

    def split_by_episode(self,):
        # split batch by way, shot and query

    def reset_base_info(self,):
        # change way, shot and query
```

+ `__init__`：init func，used to initialize the few shot learning settings like way, shot, query and other train parameters.
+ `set_forward`：used to be called in inference phase, return classifier's output and accuracy.
+ `set_forward_loss`：used to be called in training phase, return classifier's output, accuracy and loss.
+ `forward`：override the forward function `forward`  of `Module` in `pytorch`, return the ouput of `backbone`.
+ `train`：override the forward function `train` of `Module` in `pytorch`, used to unfix the `BatchNorm` layer parameter.
+ `eval`：override the forward function `test` of `Module` in `pytorch`, used to fix the `BatchNorm` layer parameter.
+ `_init_network`：used to initialize all network parameters.
+ `_generate_local_targets`：used to generate `target` for few shot learning.
+ `split_by_episode`：used to split batch in shape:[episode_size, way, shot+query, ...]. It has several split modes.
+ `reset_base_info`：used to change the few shot learning settings.

New methods must override the `set_forward` and `set_forward_loss` functions, and all other functions can be called according to the needs of the implemented methods.

Note that in order for the newly added method to be called through reflection, add a line to the `__init__.py` file in the directory of the corresponding method type:

```python
from NewMethodFileName import *
```

## metric based

Using `DN4` as an example, we will describe how to add a new `metric based classifier` to `LibFewShot`.

`metric based` methods have a common parent class `MetricModel`, which is inherited from `AbstractModel`.

```python
class MetricModel(AbstractModel):
    def __init__(self,):
        super(MetricModel, self).__init__()

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    def forward(self, x):
        out = self.emb_func(x)
        return out
```

Since the `pipeline` of `metric based` methods are mostly simple, `MetricModel` just inherites `AbstractModel` and no other changes are made.

### build model

First, create `DN4` model class, add file `dn4.py` under `core/model/metric/`: (this code have some differences with source code)

```python
class DN4(MetricModel):
    def __init__(self, n_k=3, **kwargs):
        # base info
        super(DN4Layer, self).__init__(**kwargs)
        self.n_k = n_k
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        # inference phase
        """
        :param batch: (images, labels)
        :param batch.images: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query),C,H,W]
        :param batch.labels: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query), ]
        :return: net output and accuracy
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )

        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        query_feat = query_feat.view(
            t, self.way_num * self.query_num, c, h * w
        ).permute(0, 1, 3, 2)
        query_feat = F.normalize(query_feat, p=2, dim=2).unsqueeze(2)

        # t, ws, c, h, w -> t, w, s, c, hw -> t, 1, w, c, shw
        support_feat = (
            support_feat.view(t, self.way_num, self.shot_num, c, h * w)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(t, self.way_num, c, self.shot_num * h * w)
        )
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)

        # t, wq, w, hw, shw -> t, wq, w, hw, n_k -> t, wq, w
        relation = torch.matmul(query_feat, support_feat)
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        score = torch.sum(topk_value, dim=[3, 4])

        output = score.view(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target)

        return output, acc

    def set_forward_loss(self, batch):
        # training phase
        """
        :param batch: (images, labels)
        :param batch.images: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query),C,H,W]
        :param batch.labels: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query), ]
        :return: net output, accuracy and train loss
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        emb = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            emb, mode=2
        )

        t, wq, c, h, w = query_feat.size()
        _, ws, _, _, _ = support_feat.size()

        # t, wq, c, hw -> t, wq, hw, c -> t, wq, 1, hw, c
        query_feat = query_feat.view(
            t, self.way_num * self.query_num, c, h * w
        ).permute(0, 1, 3, 2)
        query_feat = F.normalize(query_feat, p=2, dim=2).unsqueeze(2)

        # t, ws, c, h, w -> t, w, s, c, hw -> t, 1, w, c, shw
        support_feat = (
            support_feat.view(t, self.way_num, self.shot_num, c, h * w)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(t, self.way_num, c, self.shot_num * h * w)
        )
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)

        # t, wq, w, hw, shw -> t, wq, w, hw, n_k -> t, wq, w
        relation = torch.matmul(query_feat, support_feat)
        topk_value, _ = torch.topk(relation, self.n_k, dim=-1)
        score = torch.sum(topk_value, dim=[3, 4])

        output = score.view(episode_size * self.way_num * self.query_num, self.way_num)
        loss = self.loss_func(output, query_target)
        acc = accuracy(output, query_target)

        return output, acc, loss
```

`__init__` function call `super.__init__()` to initialize few shot learning settings, and initialize `DN4` method's super parameter `n_k`.

Please notice line  `19-27,65-73`, these lines aim to split batch feature vectors into correct shape that fit few shot learning setting. In deatils, in order to maximize the useage of computing resources, we first get all images' feature vectors, and then divide the feature vectors into `support set`, `suery set`. `29-50` lines are used to calculate DN4 method's output. Finally, the ouput shape of `set_forward` is $output.shape:[episode\_size*way*query,way]，acc:float$, the output shape of `set_forward_loss` is $output.shape:[episode\_size*way*query,way], acc:float, loss:tensor$. Where `output` needs to be cabculated according to the method, `acc` can call the `accuracy` function provided by `LibFewShot` and input `output, target` to get the classification accuracy.While `loss` can use the loss function that the user initializes at the start of the method, used in `set_forward_loss` to get the classification loss.

The metric based method simply needs to process the input images into the corresponding form according to the method, and then begin the training.

## meta learning

Using `MAML` as an example, we will describe how to add a new `meta learning classifier` to `LibFewShot`.

`meta learning` methods have a common parent class `MetaModel`, which is inherited from `AbstractModel`.

```python
class MetaModel(AbstractModel):
    def __init__(self,):
        super(MetaModel, self).__init__(init_type, ModelType.META, **kwargs)

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    def forward(self, x):
        out = self.emb_func(x)
        return out

    @abstractmethod
    def set_forward_adaptation(self, *args, **kwargs):
        pass

    def sub_optimizer(self, parameters, config):
        kwargs = dict()

        if config["kwargs"] is not None:
            kwargs.update(config["kwargs"])
        return getattr(torch.optim, config["name"])(parameters, **kwargs)
```

The meta-learning method adds two new functions, `set_forward_adaptation` and `sub_optimizer`. `set_forward_adaptation` is the logic that deals with the need to fine-tune the network during the classification process, and `sub_optimizer` is to provide a new sub-optimizer for the fine-tuning.

### build model

First, create `MAML` model class, add file `maml.py` under `core/model/meta/`: (this code have some differences with source code)

```python
from ..backbone.utils import convert_maml_module

class MAML(MetaModel):
    def __init__(self, inner_param, feat_dim, **kwargs):
        super(MAML, self).__init__(**kwargs)
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(nn.Linear(feat_dim, self.way_num))
        self.inner_param = inner_param

        convert_maml_module(self)

    def forward_output(self, x):
         """
        :param x: feature vectors, shape: [batch, C]
        :return: probability of classification
        """
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def set_forward(self, batch):
         """
        :param batch: (images, labels)
        :param batch.images: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query),C,H,W]
        :param batch.labels: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query), ]
        :return: net output, accuracy and train loss
        """
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        support_image, query_image, support_target, query_target = self.split_by_episode(
            image, mode=2
        )
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch):
         """
        :param batch: (images, labels)
        :param batch.images: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query),C,H,W]
        :param batch.labels: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query), ]
        :return: net output, accuracy and train loss
        """
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        support_image, query_image, support_target, query_target = self.split_by_episode(
            image, mode=2
        )
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_support_target)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, support_target):
        lr = self.inner_param["lr"]
        fast_parameters = list(self.parameters())
        for parameter in self.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()
        for i in range(self.inner_param["iter"]):
            output = self.forward_output(support_set)
            loss = self.loss_func(output, support_target)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []

            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - lr * grad[k]
                else:
                    weight.fast = weight.fast - lr * grad[k]
                fast_parameters.append(weight.fast)
```

The most important parts of `MAML `are the two parts. The first part is the `convert_maml_module `function on line `10`, which changes all the layers in the network to MAML format layers for easy parameter updating. The other part is the `set_forward_adaptation ` function, which updates the fast parameters of the network. `MAML `is a common meta learning method, so we will use MAML as an example to show how to add `meta learning ` method to `LibFewShot`.

## fine-tuning

Using `Baseline` as an example, we will describe how to add a new `fine-tuning classifier` to `LibFewShot`.

`fine-tuning` methods have a common parent class `FinetuningModel`, which is inherited from `AbstractModel`.

```python
class FinetuningModel(AbstractModel):
    def __init__(self,):
        super(FinetuningModel, self).__init__()
        # ...

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    def forward(self, x):
        out = self.emb_func(x)
        return out

    @abstractmethod
    def set_forward_adaptation(self, *args, **kwargs):
        pass

    def sub_optimizer(self, model, config):
        kwargs = dict()
        if config["kwargs"] is not None:
            kwargs.update(config["kwargs"])
        return getattr(torch.optim, config["name"])(model.parameters(), **kwargs)
```

The main aim of finetuning method train phase is to train a good feature extractor, while using the few shot learning setting in the test phase to finetune the model by the support set. Another method is to use the training setting of few shot learning to fine-tune the whole model after the feature extractor is trained. In line with the `meta learning` method, a `set_forward_adaptation` abstract function is added to handle the forward process during test phase. In addition, since there are some `fine-tuning` methods in which the classifier needs to be trained, a `sub_optimizer `method is added, passing in the parameters to be optimized and the optimized configuration parameters, and returning the optimizer for easy call.

### build model

First, create `Baseline` model class, add file `baseline.py` under `core/model/finetuning/`: (this code have some differences with source code)

```python
class Baseline(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(Baseline, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param

        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """
        :param batch: (images, labels)
        :param batch.images: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query),C,H,W]
        :param batch.labels: shape: [episodeSize*way*(shot*augment_times+query*augment_times_query), ]
        :return: net output, accuracy and train loss
        """
        image, global_target = batch
        image = image.to(self.device)
        feat = self.emb_func(image)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        episode_size = support_feat.size(0)

        support_target = support_target.reshape(episode_size, self.way_num, self.shot_num)
        query_target = query_target.reshape(episode_size, self.way_num, self.query_num)

        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(support_feat, support_target, query_feat)
            output_list.append(output)

        output = torch.stack(output_list, dim=0)
        acc = accuracy(output, query_target)

        return output, acc

    def set_forward_loss(self, batch):
        """
        :param batch: (images, labels)
        :param batch.images: shape: [batch_size,C,H,W]
        :param batch.labels: shape: [batch_size, ]
        :return: net output, accuracy and train loss
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image)
        output = self.classifier(feat)
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        """
        support_feat: shape: [way_num, shot_num, C]
        support_target: shape: [way_num*shot_num, ]
        query_feat: shape: [way_num, shot_num, C]
        """
        classifier = nn.Linear(self.feat_dim, self.way_num)
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)
        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                select_id = rand_id[i : min(i + self.inner_param["inner_batch_size"], support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]

                output = classifier(batch)

                loss = self.loss_func(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        output = classifier(query_feat)
        return output
```

The `set_forward_loss` is the same as the classical supervised classification method, while the `set_forward` is the same as the `meta learning` method. The contents of the `set_forward_adaptation `function is the main part of the test phase. The feature vectors of `support set` and `query set` extracted by backbone is used to train a classifier, and the feature vectors of `query set` is used to classify by the classifier.
