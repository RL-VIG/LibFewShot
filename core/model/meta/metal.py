import torch
import torch.nn as nn
import torch.nn.functional as F

from .meta_model import MetaModel
from ..backbone.utils import convert_maml_module
from .maml import MAMLLayer
from core.utils import accuracy


class METAL(MetaModel):

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if "norm_layer" not in name:
                    param_dict[name] = param

        return param_dict

    def __init__(self, inner_param, feat_dim, **kwargs):
        """
        inner_param:
            lr: 1e-2
            train_iter: 5
            test_iter: 10
        feat_dim: 640
        """
        super(METAL, self).__init__(**kwargs)
        # TODO feat_dim 的值有问题
        # num_classes_per_set -> way_num
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
        base_learner_num_layers = len(list(self.classifier.named_parameters()))
        support_meta_loss_num_dim = base_learner_num_layers + 2 * self.way_num + 1
        support_adapter_num_dim = base_learner_num_layers + 1
        query_num_dim = base_learner_num_layers + 1 + self.way_num

        self.loss_func = MetaLossNetwork(support_meta_loss_num_dim, inner_param)

        self.query_loss_func = MetaLossNetwork(query_num_dim, inner_param)

        self.loss_adapter = LossAdapter(support_adapter_num_dim, args=inner_param, num_loss_net_layers=2)

        self.query_loss_adapter = LossAdapter(query_num_dim, args=inner_param, num_loss_net_layers=2)

        self.feat_dim = feat_dim
        self.inner_param = inner_param
        convert_maml_module(self)

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def set_forward(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()

        output_list = []
        for i in range(episode_size):
            """
            源代码：
            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
        
            y_support_set_task = y_support_set_task.view(-1) 
            y_target_set_task = y_target_set_task.view(-1)
            """
            # 都是x
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            # 都是y
            episode_support_targets = support_target[i].reshape(-1)
            episode_query_targets = query_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_query_image, episode_support_targets,
                                        episode_query_targets)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch  # unused global_target
        image = image.to(self.device)
        (
            support_image,
            query_image,
            support_target,
            query_target,
        ) = self.split_by_episode(image, mode=2)
        episode_size, _, c, h, w = support_image.size()
        # TODO

        output_list = []
        for i in range(episode_size):
            # 都是x
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            # 都是y
            episode_support_targets = support_target[i].reshape(-1)
            episode_query_targets = query_target[i].reshape(-1)
            self.set_forward_adaptation(episode_support_image, episode_query_image, episode_support_targets,
                                        episode_query_targets)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = F.cross_entropy(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, query_set, support_target, query_target):
        lr = self.inner_param["lr"]
        fast_parameters = list(self.classifier.parameters())
        for parameter in self.classifier.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()
        for i in range(
                self.inner_param["train_iter"]
                if self.training
                else self.inner_param["test_iter"]
        ):  # num_step = i
            # adapt loss weights
            # support_set--x, query_set--x_t, support_target--y, query_target--y_t
            """
             # 都是x
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            # 都是y
            episode_support_targets = support_target[i].reshape(-1)
            episode_query_targets = query_target[i].reshape(-1)
            """
            tmp_preds = self.forward_output(x=torch.cat((support_set, query_set), 0))
            support_preds = tmp_preds[:-query_set.size(0)]
            query_preds = tmp_preds[-query_set.size(0):]
            weights = dict(self.classifier.named_parameters())  # name_param of classifier
            meta_loss_weights = dict(self.loss_func.named_parameters())  # name_param of loss_func
            meta_query_loss_weights = dict(self.query_loss_func.named_parameters())  # name_param of loss_query_func

            support_task_state = []

            support_loss = F.cross_entropy(input=support_preds, target=support_target)
            support_task_state.append(support_loss)

            for v in weights.values():
                support_task_state.append(v.mean())

            support_task_state = torch.stack(support_task_state)
            adapt_support_task_state = (support_task_state - support_task_state.mean()) / (
                    support_task_state.std() + 1e-12)

            updated_meta_loss_weights = self.loss_adapter(adapt_support_task_state, i, meta_loss_weights)

            support_y = torch.zeros(support_preds.shape).to(support_preds.device)
            support_y[torch.arange(support_y.size(0)), support_target] = 1
            support_task_state = torch.cat((
                support_task_state.view(1, -1).expand(support_preds.size(0), -1),
                support_preds,
                support_y
            ), -1)

            support_task_state = (support_task_state - support_task_state.mean()) / (support_task_state.std() + 1e-12)
            meta_support_loss = self.loss_func(support_task_state, i,
                                               params=updated_meta_loss_weights).mean().squeeze()

            query_task_state = []
            for v in weights.values():
                query_task_state.append(v.mean())
            out_prob = F.log_softmax(query_preds)
            instance_entropy = torch.sum(torch.exp(out_prob) * out_prob, dim=-1)
            query_task_state = torch.stack(query_task_state)
            query_task_state = torch.cat((
                query_task_state.view(1, -1).expand(instance_entropy.size(0), -1),
                query_preds,
                instance_entropy.view(-1, 1)
            ), -1)

            query_task_state = (query_task_state - query_task_state.mean()) / (query_task_state.std() + 1e-12)
            updated_meta_query_loss_weights = self.query_loss_adapter(query_task_state.mean(0), i,
                                                                      meta_query_loss_weights)

            meta_query_loss = self.query_loss_func(query_task_state, i,
                                                   params=updated_meta_query_loss_weights).mean().squeeze()

            loss = support_loss + meta_query_loss + meta_support_loss

            preds = support_preds
            # end
            output = self.forward_output(support_set)
            # 下面应该是 使用 loss,
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)
            fast_parameters = []

            for k, weight in enumerate(list(self.classifier.parameters())):
                if grad[k] is not None:
                    if weight.fast is None:
                        weight.fast = weight - lr * grad[k]
                    else:
                        weight.fast = weight.fast - lr * grad[k]
                    fast_parameters.append(weight.fast)


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    # print(current_dict.keys(), output_dict.keys())
    return output_dict


class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            pass
            # print('no inner loop params', self)

            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(x.shape)
        # output=input_tensor×weight_tensor^T
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MetaStepLossNetwork(nn.Module):
    def __init__(self, input_dim, args):
        super(MetaStepLossNetwork, self).__init__()

        self.args = args
        self.input_dim = input_dim
        self.input_shape = (1, input_dim)

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x

        self.linear1 = MetaLinearLayer(input_shape=self.input_shape,
                                       num_filters=self.input_dim, use_bias=True)

        self.linear2 = MetaLinearLayer(input_shape=(1, self.input_dim),
                                       num_filters=1, use_bias=True)

        out = self.linear1(out)
        out = F.relu_(out)
        out = self.linear2(out)

    def forward(self, x, params=None):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """

        linear1_params = None
        linear2_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            linear1_params = params['linear1']
            linear2_params = params['linear2']

        out = x

        out = self.linear1(out, linear1_params)
        out = F.relu_(out)
        out = self.linear2(out, linear2_params)

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


class MetaLossNetwork(nn.Module):
    def __init__(self, input_dim, args):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param input_dim: The input image batch shape.
        :param args: A named tuple containing the system's hyperparameters.
        """
        super(MetaLossNetwork, self).__init__()

        self.args = args
        self.input_dim = input_dim
        self.input_shape = (1, input_dim)

        self.num_steps = args['train_iter']  # number of inner-loop steps

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_steps):
            self.layer_dict['step{}'.format(i)] = MetaStepLossNetwork(self.input_dim, args=self.args)

            out = self.layer_dict['step{}'.format(i)](x)

    def forward(self, x, num_step, params=None):
        param_dict = dict()

        if params is not None:
            params = {key: value for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        out = self.layer_dict['step{}'.format(num_step)](out, param_dict['step{}'.format(num_step)])

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


class StepLossAdapter(nn.Module):
    def __init__(self, input_dim, num_loss_net_layers, args):
        super(StepLossAdapter, self).__init__()

        self.args = args
        output_dim = num_loss_net_layers * 2 * 2  # 2 for weight and bias, another 2 for multiplier and offset

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(input_dim, output_dim)

        self.multiplier_bias = nn.Parameter(torch.zeros(output_dim // 2))
        self.offset_bias = nn.Parameter(torch.zeros(output_dim // 2))

    def forward(self, task_state, num_step, loss_params):

        out = self.linear1(task_state)
        out = F.relu_(out)
        out = self.linear2(out)

        generated_multiplier, generated_offset = torch.chunk(out, chunks=2, dim=-1)

        i = 0
        updated_loss_weights = dict()
        for key, val in loss_params.items():
            if 'step{}'.format(num_step) in key:
                updated_loss_weights[key] = (1 + self.multiplier_bias[i] * generated_multiplier[i]) * val + \
                                            self.offset_bias[i] * generated_offset[i]
                i += 1

        return updated_loss_weights


class LossAdapter(nn.Module):
    def __init__(self, input_dim, num_loss_net_layers, args):
        super(LossAdapter, self).__init__()

        self.args = args

        self.num_steps = args['train_iter']  # number of inner-loop steps

        self.loss_adapter = nn.ModuleList()
        for i in range(self.num_steps):
            self.loss_adapter.append(StepLossAdapter(input_dim, num_loss_net_layers, args))

    def forward(self, task_state, num_step, loss_params):
        return self.loss_adapter[num_step](task_state, num_step, loss_params)
