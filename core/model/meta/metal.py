# -*- coding: utf-8 -*-
"""
@inproceedings{baik2021meta,
  title={Meta-learning with task-adaptive loss function for few-shot learning},
  author={Baik, Sungyong and Choi, Janghoon and Kim, Heewon and Cho, Dohee and Min, Jaesik and Lee, Kyoung Mu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={9465--9474},
  year={2021}
}
https://arxiv.org/abs/2110.03909

Adapted from https://github.com/baiksung/MeTAL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import accuracy
from .meta_model import MetaModel
from .maml import MAMLLayer
from ..backbone.utils import convert_maml_module

class METAL(MetaModel):

    def __init__(self, inner_param, feat_dim, **kwargs):
        super(METAL, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.classifier = MAMLLayer(feat_dim, way_num=self.way_num)
        self.inner_param = inner_param

        base_learner_num_layers = len(list(self.classifier.named_parameters()))
        support_meta_loss_num_dim = base_learner_num_layers + 2 * self.way_num + 1
        support_adapter_num_dim = base_learner_num_layers + 1
        query_num_dim = base_learner_num_layers + 1 + self.way_num
        
        self.meta_loss = MetaLossNetwork(support_meta_loss_num_dim, inner_param)
        self.meta_query_loss = MetaLossNetwork(query_num_dim, inner_param)
        self.meta_loss_adapter = LossAdapter(support_adapter_num_dim, 2, inner_param)
        self.meta_query_loss_adapter = LossAdapter(query_num_dim, 2, inner_param)

        convert_maml_module(self)

    def forward_output(self, x):
        out1 = self.emb_func(x)
        out2 = self.classifier(out1)
        return out2

    def set_forward(self, batch):
        image, _ = batch
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
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_targets = support_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_query_image, episode_support_targets)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, _ = batch
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
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_targets = support_target[i].reshape(-1)

            self.set_forward_adaptation(episode_support_image, episode_query_image, episode_support_targets)

            output = self.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        loss = F.cross_entropy(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_set, query_set, support_target):
        lr = self.inner_param["lr"]
        train_iters = self.inner_param["train_iter"] if self.training else self.inner_param["test_iter"]
    
        fast_parameters = list(self.classifier.parameters())
        for parameter in self.classifier.parameters():
            parameter.fast = None

        self.emb_func.train()
        self.classifier.train()

        for i in range(train_iters):
            combined_set = torch.cat((support_set, query_set), 0)
            tmp_preds = self.forward_output(x=combined_set)
            support_preds, query_preds = tmp_preds[:-query_set.size(0)], tmp_preds[-query_set.size(0):]

            classifier_weights = dict(self.classifier.named_parameters())
            meta_loss_weights = dict(self.meta_loss.named_parameters())
            meta_query_loss_weights = dict(self.meta_query_loss.named_parameters())

            support_loss = F.cross_entropy(input=support_preds, target=support_target)
            support_task_state = [support_loss] + [v.mean() for v in classifier_weights.values()]
            support_task_state = torch.stack(support_task_state)
            normalized_support_task_state = (support_task_state - support_task_state.mean()) / (support_task_state.std() + 1e-12)
        
            updated_meta_loss_weights = self.meta_loss_adapter(normalized_support_task_state, i, meta_loss_weights)

            support_y = torch.zeros(support_preds.shape).to(support_preds.device)
            support_y[torch.arange(support_y.size(0)), support_target] = 1
            support_task_state = torch.cat((
                normalized_support_task_state.view(1, -1).expand(support_preds.size(0), -1),
                support_preds,
                support_y
            ), -1)
            
            normalized_support_task_state = (support_task_state - support_task_state.mean()) / (support_task_state.std() + 1e-12)
            meta_support_loss = self.meta_loss(normalized_support_task_state, i, params=updated_meta_loss_weights).mean().squeeze()

            query_task_state = [v.mean() for v in classifier_weights.values()]
            log_prob_query_preds = F.log_softmax(query_preds, dim=-1)
            instance_entropy = torch.sum(torch.exp(log_prob_query_preds) * log_prob_query_preds, dim=-1)
            query_task_state = torch.stack(query_task_state)
            query_task_state = torch.cat((
                query_task_state.view(1, -1).expand(instance_entropy.size(0), -1),
                query_preds,
                instance_entropy.view(-1, 1)
            ), -1)
            
            normalized_query_task_state = (query_task_state - query_task_state.mean()) / (query_task_state.std() + 1e-12)
            updated_meta_query_loss_weights = self.meta_query_loss_adapter(normalized_query_task_state.mean(0), i, meta_query_loss_weights)

            meta_query_loss = self.meta_query_loss(normalized_query_task_state, i, params=updated_meta_query_loss_weights).mean().squeeze()

            total_loss = support_loss + meta_support_loss + meta_query_loss

            gradients = torch.autograd.grad(total_loss, fast_parameters, create_graph=True, allow_unused=True)
            fast_parameters = []
            for k, weight in enumerate(list(self.classifier.parameters())):
                if gradients[k] is not None:
                    weight.fast = weight - lr * gradients[k] if weight.fast is None else weight.fast - lr * gradients[k]
                    fast_parameters.append(weight.fast)

def extract_top_level_dict(current_dict):
    output_dict = {}

    for key, value in current_dict.items():
        name = key.replace("layer_dict.", "").replace("block_dict.", "").replace("module-", "")
        
        parts = name.split(".", 1)
        top_level = parts[0]
        sub_level = parts[1] if len(parts) > 1 else ""

        if top_level not in output_dict:
            output_dict[top_level] = value if sub_level == "" else {sub_level: value}
        else:
            if isinstance(output_dict[top_level], dict):
                output_dict[top_level][sub_level] = value
            else:
                output_dict[top_level] = {sub_level: value}

    return output_dict



class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape
        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        weight = self.weights
        bias = self.bias if self.use_bias else None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight = params["weights"]
            if self.use_bias:
                bias = params["bias"]

        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MetaStepLossNetwork(nn.Module):
    def __init__(self, input_dim, args):
        super(MetaStepLossNetwork, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.input_shape = (1, input_dim)
        self.build_network()

    def build_network(self):
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

    def restore_backup_stats(self):
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


class MetaLossNetwork(nn.Module):
    def __init__(self, input_dim, args):
        super(MetaLossNetwork, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.input_shape = (1, input_dim)
        self.num_steps = args['test_iter']
        self.build_network()

    def build_network(self):
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

        for name, _ in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        out = self.layer_dict['step{}'.format(num_step)](out, param_dict['step{}'.format(num_step)])

        return out

    def restore_backup_stats(self):
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


class StepLossAdapter(nn.Module):
    def __init__(self, input_dim, num_loss_net_layers, args):
        super(StepLossAdapter, self).__init__()

        self.args = args
        output_dim = num_loss_net_layers * 2 * 2

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
        self.num_steps = args['test_iter']
        self.loss_adapter = nn.ModuleList()
        for _ in range(self.num_steps):
            self.loss_adapter.append(StepLossAdapter(input_dim, num_loss_net_layers, args))

    def forward(self, task_state, num_step, loss_params):
        return self.loss_adapter[num_step](task_state, num_step, loss_params)