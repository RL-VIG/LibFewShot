import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


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
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


def extract_top_level_dict(current_dict):
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


class MetaStepLossNetwork(nn.Module):
    def __init__(self, input_dim, device):
        super(MetaStepLossNetwork, self).__init__()

        self.linear2 = None
        self.linear1 = None
        self.device = device
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
                if param.requires_grad:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad:
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
    def __init__(self, input_dim, device):

        super(MetaLossNetwork, self).__init__()

        self.layer_dict = None
        self.device = device
        self.input_dim = input_dim
        self.input_shape = (1, input_dim)
        # TODO 修改成配置文件 num_steps
        self.num_steps = 5

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
            self.layer_dict['step{}'.format(i)] = MetaStepLossNetwork(self.input_dim,
                                                                      device=self.device)

            out = self.layer_dict['step{}'.format(i)](x)

    def forward(self, x, num_step, params=None):
        param_dict = dict()

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
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
                if param.requires_grad:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad:
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
    def __init__(self, input_dim, num_loss_net_layers, device):
        super(StepLossAdapter, self).__init__()

        self.device = device
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
    def __init__(self, input_dim, num_loss_net_layers, device):
        super(LossAdapter, self).__init__()

        self.device = device
        # TODO 修改成配置文件 num_steps

        self.num_steps = 5  # number of inn  r-loop steps

        self.loss_adapter = nn.ModuleList()
        for i in range(self.num_steps):
            self.loss_adapter.append(StepLossAdapter(input_dim, num_loss_net_layers, device=device))

    def forward(self, task_state, num_step, loss_params):
        return self.loss_adapter[num_step](task_state, num_step, loss_params)
