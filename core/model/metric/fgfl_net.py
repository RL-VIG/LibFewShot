# -*- coding: utf-8 -*-
"""
FGFL (Frequency-aware Gain-guided Feature Learning) implementation for LibFewShot.

This module implements the complete FGFL framework preserving all core components
and mechanisms for few-shot learning with frequency guidance.

Reference:
@inproceedings{cheng2023frequency,
    title={Frequency guidance matters in few-shot learning},
    author={Cheng, Hao and Yang, Siyuan and Zhou, Joey Tianyi and Guo, Lanqing and Wen, Bihan},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={11814--11824},
    year={2023}
}

Adapted from: https://github.com/chenghao-ch94/FGFL.git
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    import torchjpeg.dct as dctt
    DCT_AVAILABLE = True
except ImportError:
    DCT_AVAILABLE = False

from core.utils import accuracy
from .metric_model import MetricModel


# ===============================
# Utility Classes and Functions
# ===============================

class _ReverseGrad(Function):
    """Custom autograd function for gradient reversal."""
    
    @staticmethod
    def forward(ctx, input, grad_scaling):
        ctx.grad_scaling = grad_scaling
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_scaling = ctx.grad_scaling
        return -grad_scaling * grad_output, None


reverse_grad = _ReverseGrad.apply


class ReverseGrad(nn.Module):
    """Gradient reversal layer.
    
    Acts as an identity layer in forward pass, but reverses the sign
    of the gradient in backward pass.
    """

    def forward(self, x, grad_scaling=1.0):
        assert grad_scaling >= 0, f"grad_scaling must be non-negative, but got {grad_scaling}"
        return reverse_grad(x, grad_scaling)


def is_bn(m):
    """Check if module is a BatchNorm layer."""
    return isinstance(m, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.BatchNorm1d))


def one_hot(indices, depth):
    """
    Returns a one-hot tensor (PyTorch equivalent of tf.one_hot).
    
    Args:
        indices: Tensor of shape (n_batch, m) or (m)
        depth: Scalar representing the depth of the one hot dimension
        
    Returns:
        One-hot tensor of shape (n_batch, m, depth) or (m, depth)
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies


def take_bn_layers(model):
    """Extract all BatchNorm layers from a model."""
    for m in model.modules():
        if is_bn(m):
            yield m
    return []


# ===============================
# Attention Mechanisms
# ===============================

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""

    def __init__(self, temperature, attn_dropout=0.1, ba=0.5):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.ba = nn.Parameter(torch.tensor(ba))

    def forward(self, q, k, v):
        # Dimension validation
        if q.size(0) != k.size(0) or q.size(0) != v.size(0):
            print(f"Batch size mismatch: q={q.shape}, k={k.shape}, v={v.shape}")

        if k.size(1) != v.size(1):
            print(f"Sequence length mismatch: k={k.shape}, v={v.shape}")

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        # Attention shape validation
        if attn.size(-1) <= 0 or attn.size(-2) <= 0:
            print(f"Invalid attention shape: {attn.shape}")
            return torch.zeros_like(q)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # Linear projections
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        # Weight initialization
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        
        # Linear projections and reshape
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention computation
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        # Apply attention
        output = self.attention(q, k, v)

        # Reshape and apply final layers
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


# ===============================
# Main FGFL Model
# ===============================

class GAINModel(MetricModel):
    """
    FGFL (Frequency-aware Gain-guided Feature Learning) Model.
    
    This model implements the FGFL framework for few-shot learning,
    incorporating frequency domain analysis and gain-guided feature learning.
    """
    
    def __init__(self, **kwargs):
        # Extract configuration parameters
        self.way_num = kwargs.get("way_num", 5)
        self.shot_num = kwargs.get("shot_num", 1)
        self.query_num = kwargs.get("query_num", 15)
        self.test_way = kwargs.get("test_way", self.way_num)
        self.test_shot = kwargs.get("test_shot", self.shot_num)
        self.test_query = kwargs.get("test_query", self.query_num)
        self.temperature = kwargs.get("temperature", 64.0)
        self.temperature2 = kwargs.get("temperature2", 64.0)
        self.use_euclidean = kwargs.get("use_euclidean", False)
        self.balance = kwargs.get("balance", 0.01)
        self.img_size = kwargs.get("image_size", 84)

        # Initialize parent class
        super(GAINModel, self).__init__(**kwargs)

        # Model architecture setup
        hdim = 640
        from ..backbone.fgfl_resnet12 import FGFLResNet

        # Dual encoders for spatial and frequency domains
        self.encoder = FGFLResNet()      # Spatial domain encoder
        self.encoder_f = FGFLResNet()    # Frequency domain encoder

        # Normalization parameters for different preprocessing
        self.mean1 = (120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0)
        self.std1 = (70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0)
        self.mean = (-1.5739, -0.8470, 0.4505)
        self.std = (66.6648, 20.2999, 18.2193)

        # Gradient computation setup
        self.grad_layer = "encoder_f.layer4"
        self.feed_forward_features = None
        self.backward_features = None
        self._register_hooks(self.grad_layer)

        # Training components
        self.reverse_layer = ReverseGrad()
        self.lambda_ = 0.5
        
        # Soft-mask parameters
        self.sigma = 0.1
        self.omega = 100
        self.temp = 12.5

        # Loss functions
        self.tri_loss_sp = nn.TripletMarginLoss(margin=0.1, p=2)

        # Batch normalization layers
        self.bn_layers_s = list(take_bn_layers(self.encoder))
        self.bn_layers_f = list(take_bn_layers(self.encoder_f))

        # Multi-head attention for feature enhancement
        self.feat_dim = hdim
        self.slf_attn2 = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    # ===============================
    # Utility Methods
    # ===============================
    
    def set_lambda(self, para):
        """Dynamically adjust the lambda parameter for gradient reversal."""
        self.lambda_ = 2.0 / (1 + math.exp(-10.0 * para)) - 1
        return self.lambda_

    def freeze_forward(self, x, freq=False):
        """Forward pass with frozen batch normalization layers."""
        bn_layers = self.bn_layers_f if freq else self.bn_layers_s
        
        is_train = len(bn_layers) > 0 and bn_layers[0].training
        if is_train:
            self._set_bn_train_status(bn_layers, is_train=False)
            
        if freq:
            _, instance_embs = self.encoder_f(x)
        else:
            _, instance_embs = self.encoder(x)

        if is_train:
            self._set_bn_train_status(bn_layers, is_train=True)

        return instance_embs

    def _set_bn_train_status(self, layers, is_train: bool):
        """Set training status for batch normalization layers."""
        for layer in layers:
            layer.train(mode=is_train)
            layer.weight.requires_grad = is_train
            layer.bias.requires_grad = is_train

    def _register_hooks(self, grad_layer):
        """Register forward and backward hooks for gradient computation."""
        def forward_hook(module, input, output):
            self.feed_forward_features = output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in self.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)
                print(f"Registered forward and backward hooks for {grad_layer}")
                gradient_layer_found = True
                break

        if not gradient_layer_found:
            raise AttributeError(f"Gradient layer {grad_layer} not found in the model")

    def _to_ohe(self, labels, num_classes=5):
        """Convert labels to one-hot encoding."""
        ohe = torch.zeros((labels.size(0), num_classes))
        for i, label in enumerate(labels):
            ohe[i, label] = 1
        ohe = torch.autograd.Variable(ohe, requires_grad=True)
        return ohe

    def split_instances(self, data):
        """Split data into support and query indices."""
        if self.training:
            support_idx = (
                torch.Tensor(np.arange(self.way_num * self.shot_num))
                .long()
                .view(1, self.shot_num, self.way_num)
            )
            query_idx = (
                torch.Tensor(np.arange(
                    self.way_num * self.shot_num,
                    self.way_num * (self.shot_num + self.query_num),
                ))
                .long()
                .view(1, self.query_num, self.way_num)
            )
        else:
            support_idx = (
                torch.Tensor(np.arange(self.test_way * self.test_shot))
                .long()
                .view(1, self.test_shot, self.test_way)
            )
            query_idx = (
                torch.Tensor(np.arange(
                    self.test_way * self.test_shot,
                    self.test_way * (self.test_shot + self.test_query),
                ))
                .long()
                .view(1, self.test_query, self.test_way)
            )
        return support_idx, query_idx

    def prepare_label(self):
        """Prepare labels for training and testing."""
        if self.training:
            label = torch.arange(self.way_num, dtype=torch.int16).repeat(self.query_num)
            label_aux = torch.arange(self.way_num, dtype=torch.int8).repeat(
                self.shot_num + self.query_num
            )
            label_shot = torch.arange(self.way_num, dtype=torch.int16).repeat(self.shot_num)
            label_q2 = torch.arange(self.way_num, dtype=torch.int16).repeat(
                self.query_num * 2
            )
        else:
            label = torch.arange(self.test_way, dtype=torch.int16).repeat(self.test_query)
            label_aux = torch.arange(self.test_way, dtype=torch.int8).repeat(
                self.test_shot + self.test_query
            )
            label_shot = torch.arange(self.test_way, dtype=torch.int16).repeat(self.test_shot)
            label_q2 = torch.arange(self.test_way, dtype=torch.int16).repeat(
                self.test_query * 2
            )

        # Convert to LongTensor
        labels = [label.type(torch.LongTensor), label_aux.type(torch.LongTensor),
                 label_shot.type(torch.LongTensor), label_q2.type(torch.LongTensor)]
        
        # Move to GPU if available
        if torch.cuda.is_available():
            labels = [lab.cuda() for lab in labels]

        return labels

    # ===============================
    # Normalization Methods
    # ===============================
    
    def denorm(self, tensor):
        """Denormalize tensor using mean and std."""
        t_mean = (
            torch.FloatTensor(self.mean)
            .view(3, 1, 1)
            .expand(3, self.img_size, self.img_size)
            .cuda(device=tensor.device)
        )
        t_std = (
            torch.FloatTensor(self.std)
            .view(3, 1, 1)
            .expand(3, self.img_size, self.img_size)
            .cuda(device=tensor.device)
        )
        return tensor * t_std + t_mean

    def fnorm(self, tensor):
        """Normalize tensor using mean and std."""
        t_mean = (
            torch.FloatTensor(self.mean)
            .view(3, 1, 1)
            .expand(3, self.img_size, self.img_size)
            .cuda(device=tensor.device)
        )
        t_std = (
            torch.FloatTensor(self.std)
            .view(3, 1, 1)
            .expand(3, self.img_size, self.img_size)
            .cuda(device=tensor.device)
        )
        return (tensor - t_mean) / t_std

    def denorms(self, tensor):
        """Denormalize tensor using mean1 and std1."""
        t_mean = (
            torch.FloatTensor(self.mean1)
            .view(3, 1, 1)
            .expand(3, self.img_size, self.img_size)
            .cuda(device=tensor.device)
        )
        t_std = (
            torch.FloatTensor(self.std1)
            .view(3, 1, 1)
            .expand(3, self.img_size, self.img_size)
            .cuda(device=tensor.device)
        )
        return tensor * t_std + t_mean

    def fnorms(self, tensor):
        """Normalize tensor using mean1 and std1."""
        t_mean = (
            torch.FloatTensor(self.mean1)
            .view(3, 1, 1)
            .expand(3, self.img_size, self.img_size)
            .cuda(device=tensor.device)
        )
        t_std = (
            torch.FloatTensor(self.std1)
            .view(3, 1, 1)
            .expand(3, self.img_size, self.img_size)
            .cuda(device=tensor.device)
        )
        return (tensor - t_mean) / t_std

    # ===============================
    # Main Forward Pass Methods
    # ===============================
    
    def set_forward(self, x, **kwargs):
        """Forward pass for inference."""
        q_lab, labels, s_lab, label_q2 = self.prepare_label()
        support_idx, query_idx = self.split_instances(x)
        
        _, instance_embs = self.encoder(x)
        x2 = dctt.images_to_batch(self.denorms(x).clamp(0, 1))
        logits = self.semi_protofeat(instance_embs, support_idx, query_idx)

        return logits, accuracy(logits, q_lab)

    def set_forward_loss(self, x, **kwargs):
        """Forward pass for training with loss computation."""
        q_lab, labels, s_lab, label_q2 = self.prepare_label()
        support_idx, query_idx = self.split_instances(x)
        
        _, instance_embs = self.encoder(x)
        x2 = dctt.images_to_batch(self.denorms(x).clamp(0, 1))
        logits, logits_reg = self.semi_protofeat(instance_embs, support_idx, query_idx)

        # Generate frequency mask and associated logits
        freq_mask_results = self.gen_freq_mask(x2, s_lab, logits, q_lab)
        if len(freq_mask_results) == 4:
            logits_fsf, freq_mask, logits_am, logits_am2 = freq_mask_results
        else:
            logits_fsf, freq_mask = freq_mask_results
            logits_am = logits_am2 = None

        # Apply frequency masks
        mask_x = self.fnorms(
            dctt.batch_to_images((x2 * freq_mask).clamp(-128.0, 127.0)).clamp(0, 1)
        )
        bad_x = self.fnorms(
            dctt.batch_to_images((x2 * (1 - freq_mask)).clamp(-128.0, 127.0)).clamp(0, 1)
        )

        # Enhanced embeddings
        instance_embs_good = self.freeze_forward(mask_x, freq=False)
        instance_embs_bad = self.freeze_forward(bad_x, freq=False)

        # Triplet loss for spatial-wise enhancement
        loss_contr = self.tri_loss_sp(instance_embs, instance_embs_good, instance_embs_bad)

        # Gradient reversal for bad features
        instance_embs_bad = self.reverse_layer(instance_embs_bad.detach(), 0.1)

        # Enhanced prototypical networks
        logits_up = self.semi_protofeat_enh(
            instance_embs_good, instance_embs, support_idx, query_idx
        )
        logits_sm, logits_qm = self.contrast_bad2(
            instance_embs_bad, instance_embs, support_idx, query_idx
        )

        # Calculate losses
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, q_lab)
        loss2 = criterion(logits_fsf, labels)
        loss3 = criterion(logits_up, label_q2)
        loss_sm = criterion(logits_sm, q_lab)
        loss_qm = criterion(logits_qm, q_lab)
        
        total_loss = (
            loss + 1.0 * loss2 + 0.01 * loss3 + 0.01 * loss_contr +
            0.1 * (loss_sm + loss_qm)
        )
        
        # Add auxiliary losses if available
        if logits_am is not None and logits_am2 is not None:
            loss_am = criterion(logits_am, labels)
            loss_am2 = criterion(logits_am2, labels)
            total_loss += 0.1 * (loss_am + loss_am2)
            
        if logits_reg is not None:
            total_loss += self.balance * F.cross_entropy(logits_reg, labels)

        acc = accuracy(logits, q_lab)
        return logits, acc, total_loss

    def loss_fn_kd(self, outputs, labels, teacher_outputs, alpha=0.5, T=64, **kwargs):
        """
        Compute knowledge distillation loss.
        
        Args:
            outputs: Student model outputs
            labels: Ground truth labels
            teacher_outputs: Teacher model outputs
            alpha: Weight for KD loss vs CE loss
            T: Temperature for softmax
        """
        KD_loss = nn.KLDivLoss()(
            F.log_softmax(outputs * self.temperature / T, dim=1),
            F.softmax(teacher_outputs * self.temperature2 / T, dim=1),
        ) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1.0 - alpha)

        return KD_loss


    def gen_freq_mask(self, x, s_lab, probs, labels=None):
        """Generate frequency mask for spatial-frequency learning."""
        _, instance_embsf = self.encoder_f(x)
        support_idx, query_idx = self.split_instances(x)

        with torch.enable_grad():
            self.encoder_f.zero_grad()

            logits_fsf = self.fproto_forward2(
                instance_embsf, support_idx, query_idx, self.temp
            )

            if self.training:
                q_ohe = self._to_ohe(s_lab, self.way_num).cuda()
                q_lab = self._to_ohe(labels, self.way_num).cuda()
                q_ohe = torch.cat([q_ohe, q_lab], dim=0)
            else:
                q_ohe = self._to_ohe(s_lab, self.test_way).cuda()
                q_ohe = torch.cat([q_ohe, probs.softmax(1)], dim=0)

            gradient_q = (logits_fsf * self.temp * q_ohe).sum(dim=1)
            gradient_q.backward(gradient=torch.ones_like(gradient_q), retain_graph=True)
            self.encoder_f.zero_grad()

        backward_features = self.backward_features
        
        # Gain computation on feature map (frequency domain)
        if self.feed_forward_features is not None:
            fl = self.feed_forward_features.to(x.device)
            weights = F.adaptive_avg_pool2d(backward_features, 1).to(x.device)

            Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
            Ac = F.relu(Ac, inplace=True)
            Ac = F.interpolate(
                Ac, mode="bilinear", align_corners=True, size=(x.shape[-2], x.shape[-1])
            )

            Ac_min = Ac.min()
            Ac_max = Ac.max()
            scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min + 1e-8)
            mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma)).to(x.device)
        else:
            # Fallback: create a default mask if features are not available
            mask = torch.ones_like(x[:, :1]).to(x.device)

        if self.training:
            mask_embs = self.freeze_forward(x * (1 - mask), freq=True)
            mask_embs = self.reverse_layer(mask_embs.detach(), self.lambda_)

            logits_am, logits_am2 = self.fproto_forward_pare(
                mask_embs, instance_embsf, support_idx, query_idx, self.temp
            )

            return logits_fsf, mask, logits_am, logits_am2
        else:
            return logits_fsf, mask

    # ===============================
    # Prototypical Network Methods
    # ===============================
    
    def forward_enhanced(self, instance_good, instance_embs, support_idx, query_idx, **kwargs):
        """Enhanced forward pass with good and original instances."""

        emb_dim = instance_embs.size(-1)
        # organize support/query data
        support = (
            instance_embs[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query = (
            instance_embs[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        support_good = (
            instance_good[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query_good = (
            instance_good[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        query = torch.cat([query, query_good], dim=0)  # T x (K+Kq) x N x d
        support = torch.cat([support, support_good], dim=1)  # T x (K+Kq) x N x d

        num_batch = support.shape[0]
        num_proto = self.way_num

        aux_task = torch.cat(
            [
                support.view(1, self.shot_num * 2, self.way_num, emb_dim),
                query.view(1, self.query_num * 2, self.way_num, emb_dim),
            ],
            1,
        )  # T x (K+Kq) x N x d
        num_query = np.prod(aux_task.shape[1:3])
        aux_task = aux_task.permute([0, 2, 1, 3])
        aux_task = aux_task.contiguous().view(
            -1, self.shot_num * 2 + self.query_num * 2, emb_dim
        )

        aux_emb = self.slf_attn2(aux_task, aux_task, aux_task)  # T x N x (K+Kq) x d
        # compute class mean
        aux_emb = aux_emb.view(
            num_batch, self.way_num, self.shot_num * 2 + self.query_num * 2, emb_dim
        )
        aux_center = torch.mean(aux_emb, 2)  # T x N x d

        if self.training:

            if self.use_euclidean:
                aux_task = (
                    aux_task.permute([1, 0, 2])
                    .contiguous()
                    .view(-1, emb_dim)
                    .unsqueeze(1)
                )  # (Nbatch*Nq*Nw, 1, d)
                aux_center = (
                    aux_center.unsqueeze(1)
                    .expand(num_batch, num_query, num_proto, emb_dim)
                    .contiguous()
                )
                aux_center = aux_center.view(
                    num_batch * num_query, num_proto, emb_dim
                )  # (Nbatch x Nq, Nk, d)
                logits_reg = (
                    -torch.sum((aux_center - aux_task) ** 2, 2) / self.temperature2
                )

            else:
                aux_center = F.normalize(
                    aux_center, dim=-1
                )  # normalize for cosine distance
                aux_task = (
                    aux_task.permute([1, 0, 2])
                    .contiguous()
                    .view(num_batch, -1, emb_dim)
                )  # (Nbatch,  Nq*Nw, d)
                aux_task = F.normalize(
                    aux_task, dim=-1
                )  # normalize for cosine distance

                logits_reg = (
                    torch.bmm(aux_task, aux_center.permute([0, 2, 1]))
                    / self.temperature2
                )
                logits_reg = logits_reg.view(-1, num_proto)

            return logits_reg

        else:
            if self.use_euclidean:
                aux_task = aux_task[:, self.shot_num * 2 :, :]
                aux_task = (
                    aux_task.permute([1, 0, 2])
                    .contiguous()
                    .view(-1, emb_dim)
                    .unsqueeze(1)
                )  # (Nbatch*Nq*Nw, 1, d)
                aux_center = (
                    aux_center.unsqueeze(1)
                    .expand(num_batch, aux_task.shape[0], num_proto, emb_dim)
                    .contiguous()
                )
                aux_center = aux_center.view(
                    num_batch * aux_task.shape[0], num_proto, emb_dim
                )  # (Nbatch x Nq, Nk, d)
                logits_reg = (
                    -torch.sum((aux_center - aux_task) ** 2, 2) / self.temperature2
                )

            else:
                aux_center = F.normalize(
                    aux_center, dim=-1
                )  # normalize for cosine distance
                aux_task = aux_task[:, self.shot_num * 2 :, :]
                aux_task = (
                    aux_task.permute([1, 0, 2])
                    .contiguous()
                    .view(num_batch, -1, emb_dim)
                )  # (Nbatch,  Nq*Nw, d)
                aux_task = F.normalize(
                    aux_task, dim=-1
                )  # normalize for cosine distance

                logits_reg = (
                    torch.bmm(aux_task, aux_center.permute([0, 2, 1]))
                    / self.temperature2
                )
                logits_reg = logits_reg.view(-1, num_proto)

            return logits_reg

    def contrast_bad2(self, instance_embs, instance_embs_ori, support_idx, query_idx):
        """Contrastive learning between bad and original embeddings."""
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support_mask = (
            instance_embs[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query_mask = (
            instance_embs[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        support = (
            instance_embs_ori[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query = (
            instance_embs_ori[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        # get mean of the support
        # proto_mask = support_mask.mean(dim=1) # Ntask x NK x d
        # proto = support.mean(dim=1) # Ntask x NK x d

        num_batch = support.shape[0]
        num_query = np.prod(query_idx.shape[-2:])
        num_shot, num_way = support.shape[1], support.shape[2]

        whole_set = torch.cat(
            [support.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)],
            1,
        )
        # support = self.slf_attn2(support.view(num_batch, -1, emb_dim), whole_set, whole_set).view(num_batch, num_shot, num_way, emb_dim)
        # print("whole_set shape:", whole_set.shape)
        # print("support_idx shape:", support_idx.shape, "values:", support_idx)
        # print("query_idx shape:", query_idx.shape, "values:", query_idx)
        whole_set = self.slf_attn2(whole_set, whole_set, whole_set)
        support, query = whole_set.split([num_shot * num_way, num_query], 1)
        support = support.view(num_batch, num_shot, num_way, emb_dim)
        query = query.view(num_batch, -1, num_way, emb_dim)

        # get mean of the support
        proto = self.get_proto(
            support, query
        )  # we can also use adapted query set here to achieve better results
        # proto = support.mean(dim=1) # Ntask x NK x d
        num_proto = proto.shape[1]

        whole_set_m = torch.cat(
            [
                support_mask.view(num_batch, -1, emb_dim),
                query_mask.view(num_batch, -1, emb_dim),
            ],
            1,
        )
        # support = self.slf_attn2(support.view(num_batch, -1, emb_dim), whole_set, whole_set).view(num_batch, num_shot, num_way, emb_dim)

        whole_set_m = self.slf_attn2(whole_set_m, whole_set_m, whole_set_m)
        support_mask, query_mask = whole_set_m.split([num_shot * num_way, num_query], 1)
        support_mask = support_mask.view(num_batch, num_shot, num_way, emb_dim)
        query_mask = query_mask.view(num_batch, -1, num_way, emb_dim)

        # get mean of the support
        proto_mask = self.get_proto(
            support_mask, query_mask
        )  # we can also use adapted query set here to achieve better results
        # proto_mask = proto_mask.mean(dim=1) # Ntask x NK x d

        if self.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            query_mask = query_mask.view(-1, emb_dim).unsqueeze(
                1
            )  # (Nbatch*Nq*Nw, 1, d)

            logits = -torch.sum((proto_mask - query) ** 2, 2) / self.temperature
            logits2 = -torch.sum((proto - query_mask) ** 2, 2) / self.temperature

        else:  # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            proto_mask = F.normalize(
                proto_mask, dim=-1
            )  # normalize for cosine distance
            print(query.shape, proto.shape, proto_mask.shape)
            logits = torch.bmm(query, proto_mask.permute([0, 2, 1])) / self.temperature
            logits = logits.view(-1, num_proto)

            logits2 = torch.bmm(query_mask, proto.permute([0, 2, 1])) / self.temperature
            logits2 = logits2.view(-1, num_proto)

        return logits, logits2

    def featstar(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = (
            instance_embs[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query = (
            instance_embs[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        query = query.view(-1, emb_dim).unsqueeze(1)

        proto = (
            proto.unsqueeze(1)
            .expand(num_batch, num_query, num_proto, emb_dim)
            .contiguous()
        )
        proto = proto.view(num_batch * num_query, num_proto, emb_dim)

        # refine by Transformer
        combined = torch.cat([proto, query], 1)  # Nk x (N + 1) x d, batch_size = NK
        combined = self.slf_attn2(combined, combined, combined)
        # compute distance for all batches
        proto, query = combined.split(num_proto, 1)

        if self.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)

            logits = -torch.sum((proto - query) ** 2, 2) / self.temperature
        else:  # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance

            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.temperature
            logits = logits.view(-1, num_proto)

        if self.training:
            return logits, None
        else:
            return logits

    def get_proto(self, x_shot, x_pool):
        # get the prototypes based w/ an unlabeled pool set
        num_batch, num_shot, num_way, emb_dim = x_shot.shape
        num_pool_shot = x_pool.shape[1]
        num_pool = num_pool_shot * num_way
        label_support = torch.arange(num_way).repeat(num_shot).type(torch.LongTensor)
        label_support_onehot = one_hot(label_support, num_way)
        label_support_onehot = label_support_onehot.unsqueeze(0).repeat(
            [num_batch, 1, 1]
        )
        if torch.cuda.is_available():
            label_support_onehot = label_support_onehot.cuda()

        proto_shot = x_shot.mean(dim=1)
        if self.use_euclidean:
            dis = (
                -torch.sum(
                    (
                        proto_shot.unsqueeze(1)
                        .expand(num_batch, num_pool, num_way, emb_dim)
                        .contiguous()
                        .view(num_batch * num_pool, num_way, emb_dim)
                        - x_pool.view(-1, emb_dim).unsqueeze(1)
                    )
                    ** 2,
                    2,
                )
                / self.temperature
            )
        else:
            dis = (
                torch.bmm(
                    x_pool.view(num_batch, -1, emb_dim),
                    F.normalize(proto_shot, dim=-1).permute([0, 2, 1]),
                )
                / self.temperature
            )

        dis = dis.view(num_batch, -1, num_way)
        z_hat = F.softmax(dis, dim=2)
        z = torch.cat(
            [label_support_onehot, z_hat], dim=1
        )  # (num_batch, n_shot + n_pool, n_way)
        h = torch.cat(
            [x_shot.view(num_batch, -1, emb_dim), x_pool.view(num_batch, -1, emb_dim)],
            dim=1,
        )  # (num_batch, n_shot + n_pool, n_embedding)

        proto = torch.bmm(z.permute([0, 2, 1]), h)
        sum_z = z.sum(dim=1).view((num_batch, -1, 1))
        proto = proto / sum_z
        return proto

    def semi_protofeat(self, instance_embs, support_idx, query_idx):
        """Semi-prototypical feature learning with attention mechanism."""
        emb_dim = instance_embs.size(-1)
        batch_size = instance_embs.size(0)

        # 添加边界检查
        max_support_idx = support_idx.max().item() if support_idx.numel() > 0 else -1
        max_query_idx = query_idx.max().item() if query_idx.numel() > 0 else -1

        if max_support_idx >= batch_size:
            print(
                f"Error: support_idx max ({max_support_idx}) >= batch_size ({batch_size})"
            )
            support_idx = support_idx.clamp(0, batch_size - 1)

        if max_query_idx >= batch_size:
            print(
                f"Error: query_idx max ({max_query_idx}) >= batch_size ({batch_size})"
            )
            query_idx = query_idx.clamp(0, batch_size - 1)

        # organize support/query data
        support = (
            instance_embs[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query = (
            instance_embs[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        num_batch = support.shape[0]
        num_shot, num_way = support.shape[1], support.shape[2]
        num_query = np.prod(query_idx.shape[-2:])

        # transformation
        whole_set = torch.cat(
            [support.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)],
            1,
        )
        # support = self.slf_attn2(support.view(num_batch, -1, emb_dim), whole_set, whole_set).view(num_batch, num_shot, num_way, emb_dim)

        # 添加维度检查
        if whole_set.size(1) == 0:
            print("Warning: whole_set has 0 elements in dimension 1")
            return torch.zeros(num_batch, num_query, num_way), torch.zeros(
                num_batch, num_query, num_way
            )

        whole_set = self.slf_attn2(whole_set, whole_set, whole_set)
        support, query = whole_set.split([num_shot * num_way, num_query], 1)
        support = support.view(num_batch, num_shot, num_way, emb_dim)
        query = query.view(num_batch, -1, num_way, emb_dim)

        # get mean of the support
        proto = self.get_proto(
            support, query
        )  # we can also use adapted query set here to achieve better results
        # proto = support.mean(dim=1) # Ntask x NK x d
        num_proto = proto.shape[1]

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = (
                proto.unsqueeze(1)
                .expand(num_batch, num_query, num_proto, emb_dim)
                .contiguous()
            )
            proto = proto.view(
                num_batch * num_query, num_proto, emb_dim
            )  # (Nbatch x Nq, Nk, d)

            logits = -torch.sum((proto - query) ** 2, 2) / self.temperature
        else:
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.temperature
            logits = logits.view(-1, num_proto)

        # for regularization
        if self.training:
            aux_task = torch.cat(
                [
                    support.view(1, self.shot_num, self.way_num, emb_dim),
                    query.view(1, self.query_num, self.way_num, emb_dim),
                ],
                1,
            )  # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(
                -1, self.shot_num + self.query_num, emb_dim
            )
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn2(aux_task, aux_task, aux_task)  # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(
                num_batch, self.way_num, self.shot_num + self.query_num, emb_dim
            )
            aux_center = torch.mean(aux_emb, 2)  # T x N x d

            if self.use_euclidean:
                aux_task = (
                    aux_task.permute([1, 0, 2])
                    .contiguous()
                    .view(-1, emb_dim)
                    .unsqueeze(1)
                )  # (Nbatch*Nq*Nw, 1, d)
                aux_center = (
                    aux_center.unsqueeze(1)
                    .expand(num_batch, num_query, num_proto, emb_dim)
                    .contiguous()
                )
                aux_center = aux_center.view(
                    num_batch * num_query, num_proto, emb_dim
                )  # (Nbatch x Nq, Nk, d)

                logits_reg = (
                    -torch.sum((aux_center - aux_task) ** 2, 2) / self.temperature2
                )
            else:
                aux_center = F.normalize(
                    aux_center, dim=-1
                )  # normalize for cosine distance
                aux_task = (
                    aux_task.permute([1, 0, 2])
                    .contiguous()
                    .view(num_batch, -1, emb_dim)
                )  # (Nbatch,  Nq*Nw, d)

                logits_reg = (
                    torch.bmm(aux_task, aux_center.permute([0, 2, 1]))
                    / self.temperature2
                )
                logits_reg = logits_reg.view(-1, num_proto)

            return logits, logits_reg
            # return logits, None
        else:
            return logits

    def semi_protofeat_enh(self, instance_good, instance_embs, support_idx, query_idx):

        emb_dim = instance_embs.size(-1)
        # organize support/query data
        support = (
            instance_embs[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query = (
            instance_embs[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        support_good = (
            instance_good[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query_good = (
            instance_good[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        # support_new = torch.cat((support, support_good), dim=1)
        # proto = support_new.mean(dim=1) # Ntask x NK x d

        query = torch.cat([query, query_good], dim=1)
        # print(query.shape) # num_batch, 15*2, 5, emb_dim
        sup_enh = torch.cat([support, support_good], dim=1)
        # print(sup_enh.shape) # num_batch, 5*2, 5, emb_dim

        num_batch = support.shape[0]
        num_shot, num_way = sup_enh.shape[1], sup_enh.shape[2]
        num_query = np.prod(query_idx.shape[-2:]) * 2

        # whole_set = torch.cat([sup_enh.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)], 1)
        # support = self.slf_attn2(sup_enh.view(num_batch, -1, emb_dim), whole_set, whole_set).view(num_batch, num_shot, num_way, emb_dim)

        whole_set = torch.cat(
            [sup_enh.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)],
            1,
        )
        # if self.training:
        whole_set = self.slf_attn2(
            whole_set, whole_set, whole_set
        )  # .view(num_batch, num_shot, num_way, emb_dim)
        # else: # to-do: test
        #     whole_set = self.slf_attn2(whole_set, sup_enh.view(num_batch, -1, emb_dim), sup_enh.view(num_batch, -1, emb_dim))

        # print(whole_set.shape) # num_batch, 200, emb_dim

        support, query = whole_set.split([num_shot * num_way, num_query], 1)
        support = support.view(num_batch, num_shot, num_way, emb_dim)

        support, support_good = support.split([num_shot // 2, num_shot // 2], 1)

        query = query.view(num_batch, -1, num_way, emb_dim)

        query_enh = torch.cat([support_good, query], 1)

        # get mean of the support
        proto = self.get_proto(
            support, query_enh
        )  # we can also use adapted query set here to achieve better results
        num_proto = proto.shape[1]

        if self.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = (
                proto.unsqueeze(1)
                .expand(num_batch, num_query, num_proto, emb_dim)
                .contiguous()
            )
            proto = proto.view(
                num_batch * num_query, num_proto, emb_dim
            )  # (Nbatch x Nq, Nk, d)

            logits = -torch.sum((proto - query) ** 2, 2) / self.temperature
        else:
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.temperature
            logits = logits.view(-1, num_proto)

        return logits

    def semi_protofeat_enh_val(
        self, instance_good, instance_embs, support_idx, query_idx
    ):

        emb_dim = instance_embs.size(-1)
        # organize support/query data
        support = (
            instance_embs[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query = (
            instance_embs[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        support_good = (
            instance_good[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query_good = (
            instance_good[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        # support_new = torch.cat((support, support_good), dim=1)
        # proto = support_new.mean(dim=1) # Ntask x NK x d

        # query = torch.cat([query, query_good], dim=1)

        if self.training:
            query = torch.cat([query, query_good], dim=0).mean(dim=0)

        # print(query.shape) # num_batch, 15*2, 5, emb_dim
        sup_enh = torch.cat([support, support_good], dim=1)
        # print(sup_enh.shape) # num_batch, 5*2, 5, emb_dim

        # sup_enh = support_good

        num_batch = support.shape[0]
        num_shot, num_way = sup_enh.shape[1], sup_enh.shape[2]
        num_query = np.prod(query_idx.shape[-2:])  # *2

        ##############################

        whole_set = torch.cat(
            [sup_enh.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)],
            1,
        )
        whole_set = self.slf_attn2(whole_set, whole_set, whole_set)

        support, query = whole_set.split([num_shot * num_way, num_query], 1)
        support_enh = support.view(num_batch, num_shot, num_way, emb_dim)

        support, support_aug = support_enh.split([num_shot // 2, num_shot // 2], 1)
        support = support.view(num_batch, num_shot // 2, num_way, emb_dim)
        support_aug = support_aug.view(num_batch, num_shot // 2, num_way, emb_dim)

        # # whole_set = whole_set.view(num_batch, num_shot, num_way, emb_dim)

        proto = self.get_proto(support, support_aug)

        # proto = torch.mean(sup_enh, dim=1)
        # proto = self.slf_attn2(proto, proto, proto)

        #################################################

        num_proto = proto.shape[1]

        if self.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = (
                proto.unsqueeze(1)
                .expand(num_batch, num_query, num_proto, emb_dim)
                .contiguous()
            )
            proto = proto.view(
                num_batch * num_query, num_proto, emb_dim
            )  # (Nbatch x Nq, Nk, d)
            logits = -torch.sum((proto - query) ** 2, 2) / self.temperature
        else:
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.temperature
            logits = logits.view(-1, num_proto)

        return logits

    def semi_protofeat_enh_val2(
        self, instance_good, instance_embs, support_idx, query_idx
    ):

        emb_dim = instance_embs.size(-1)
        # organize support/query data
        support = (
            instance_embs[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query = (
            instance_embs[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        support_good = (
            instance_good[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )

        proto = support.mean(dim=1)  # Ntask x NK x d

        # query = torch.cat([query, query_good], dim=0)
        # print(query.shape) # num_batch*2, 15, 5, emb_dim
        sup_enh = torch.cat([support, support_good], dim=1)
        # print(sup_enh.shape) # num_batch, 5*2, 5, emb_dim

        num_shot, num_way = support.shape[1], support.shape[2]

        query = query.view(query.shape[0], -1, emb_dim)  # num_batch*2, 15*5, emb_dim

        sup_enh = sup_enh.expand(
            query.shape[1], -1, -1, -1
        )  # 15*5, 5 or 1 *2, 5, emb_dim

        whole_set = torch.cat(
            [sup_enh.view(sup_enh.shape[0], -1, emb_dim), query.transpose(0, 1)], 1
        )  # 15*5, num_batch*(5*(1 or 5)*2+1), emb_dim

        whole_set = self.slf_attn2(whole_set, whole_set, whole_set)

        support, support_good, query = whole_set.split(
            [num_shot * num_way, num_shot * num_way, query.shape[0]], 1
        )  # 15*5, num_batch*(5*2), emb_dim ;  15*5, 2, emb_dim

        support = support.contiguous().view(-1, num_shot, num_way, emb_dim)

        support_good = support_good.contiguous().view(-1, num_shot, num_way, emb_dim)
        proto = self.get_proto(support, support_good)

        query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
        logits = -torch.sum((proto - query) ** 2, 2) / self.temperature

        return logits

    def feat_forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = (
            instance_embs[support_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(support_idx.shape + (-1,)))
        )
        query = (
            instance_embs[query_idx.contiguous().view(-1)]
            .contiguous()
            .view(*(query_idx.shape + (-1,)))
        )

        num_batch = support.shape[0]
        num_shot, num_way = support.shape[1], support.shape[2]
        num_query = np.prod(query_idx.shape[-2:])

        proto = support.mean(dim=1)  # Ntask x NK x d
        proto = self.slf_attn2(proto, proto, proto)
        num_proto = proto.shape[1]

        if self.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = (
                proto.unsqueeze(1)
                .expand(num_batch, num_query, num_proto, emb_dim)
                .contiguous()
            )
            proto = proto.view(
                num_batch * num_query, num_proto, emb_dim
            )  # (Nbatch x Nq, Nk, d)

            logits = -torch.sum((proto - query) ** 2, 2) / self.temperature
        else:
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.temperature
            logits = logits.view(-1, num_proto)

        # for regularization
        if self.training:
            aux_task = torch.cat(
                [
                    support.view(1, self.shot_num, self.way_num, emb_dim),
                    query.view(1, self.query_num, self.way_num, emb_dim),
                ],
                1,
            )  # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(
                -1, self.shot_num + self.query_num, emb_dim
            )
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn2(aux_task, aux_task, aux_task)  # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(
                num_batch, self.way_num, self.shot_num + self.query_num, emb_dim
            )
            aux_center = torch.mean(aux_emb, 2)  # T x N x d

            if self.use_euclidean:
                aux_task = (
                    aux_task.permute([1, 0, 2])
                    .contiguous()
                    .view(-1, emb_dim)
                    .unsqueeze(1)
                )  # (Nbatch*Nq*Nw, 1, d)
                aux_center = (
                    aux_center.unsqueeze(1)
                    .expand(num_batch, num_query, num_proto, emb_dim)
                    .contiguous()
                )
                aux_center = aux_center.view(
                    num_batch * num_query, num_proto, emb_dim
                )  # (Nbatch x Nq, Nk, d)

                logits_reg = (
                    -torch.sum((aux_center - aux_task) ** 2, 2) / self.temperature2
                )
            else:
                aux_center = F.normalize(
                    aux_center, dim=-1
                )  # normalize for cosine distance
                aux_task = (
                    aux_task.permute([1, 0, 2])
                    .contiguous()
                    .view(num_batch, -1, emb_dim)
                )  # (Nbatch,  Nq*Nw, d)

                logits_reg = (
                    torch.bmm(aux_task, aux_center.permute([0, 2, 1]))
                    / self.temperature2
                )
                logits_reg = logits_reg.view(-1, num_proto)

            return logits, logits_reg

        else:
            return logits

    # ===============================
    # Frequency Domain Methods
    # ===============================
    
    def fproto_forward2(self, instance_embs, support_idx, query_idx, temp=64.0):
        """Frequency domain prototypical forward pass."""
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(
            *(support_idx.shape + (-1,))
        )
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1)  # .detach() # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]

        aux_task = torch.cat(
            [
                support.view(1, self.shot_num, self.way_num, emb_dim),
                query.view(1, self.query_num, self.way_num, emb_dim),
            ],
            1,
        )  # T x (K+Kq) x N x d
        aux_task = aux_task.permute([0, 2, 1, 3])
        aux_task = aux_task.contiguous().view(
            -1, self.shot_num + self.query_num, emb_dim
        )
        aux_task = aux_task.permute([1, 0, 2]).contiguous().view(num_batch, -1, emb_dim)

        proto = F.normalize(proto, dim=-1)
        # aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim)
        # aux_task = instance_embs.view(num_batch, -1, emb_dim)
        # aux_task = F.normalize(aux_task, dim=-1)
        logits = (
            torch.bmm(aux_task, proto.permute([0, 2, 1])) / temp
        )  # / self.temperature
        logits = logits.view(-1, num_proto)

        return logits

    def fproto_forward_pare(
        self, instance_embs_bad, instance_embs, support_idx, query_idx, temp=64.0
    ):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(
            *(support_idx.shape + (-1,))
        )
        support_bad = instance_embs_bad[support_idx.flatten()].view(
            *(support_idx.shape + (-1,))
        )

        # # get mean of the support
        # proto = support.mean(dim=1) #.detach() # Ntask x NK x d
        # num_batch = proto.shape[0]
        # num_proto = proto.shape[1]

        # # proto = self.slf_attnf(proto, proto, proto)

        # proto = F.normalize(proto, dim=-1) # normalize for cosine distance
        # query = instance_embs_bad.unsqueeze(0)

        # # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        # logits = torch.bmm(query, proto.permute([0,2,1])) / self.temperature
        # logits = logits.view(-1, num_proto)

        # get mean of the support
        proto_bad = support_bad.mean(dim=1)  # .detach() # Ntask x NK x d
        proto = support.mean(dim=1)  # .detach() # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]

        query_bad = instance_embs_bad[query_idx.flatten()].view(
            *(query_idx.shape + (-1,))
        )
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

        aux_task_g = torch.cat(
            [
                support.view(1, self.shot_num, self.way_num, emb_dim),
                query.view(1, self.query_num, self.way_num, emb_dim),
            ],
            1,
        )  # T x (K+Kq) x N x d
        aux_task_g = aux_task_g.permute([0, 2, 1, 3])
        aux_task_g = aux_task_g.contiguous().view(
            -1, self.shot_num + self.query_num, emb_dim
        )
        aux_task_g = (
            aux_task_g.permute([1, 0, 2]).contiguous().view(num_batch, -1, emb_dim)
        )

        aux_task_b = torch.cat(
            [
                support_bad.view(1, self.shot_num, self.way_num, emb_dim),
                query_bad.view(1, self.query_num, self.way_num, emb_dim),
            ],
            1,
        )  # T x (K+Kq) x N x d
        aux_task_b = aux_task_b.permute([0, 2, 1, 3])
        aux_task_b = aux_task_b.contiguous().view(
            -1, self.shot_num + self.query_num, emb_dim
        )
        aux_task_b = (
            aux_task_b.permute([1, 0, 2]).contiguous().view(num_batch, -1, emb_dim)
        )

        proto_bad = F.normalize(proto_bad, dim=-1)  # normalize for cosine distance
        # query = instance_embs.unsqueeze(0)
        # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        logits = (
            torch.bmm(aux_task_g, proto_bad.permute([0, 2, 1])) / temp
        )
        logits = logits.view(-1, num_proto)

        proto = F.normalize(proto, dim=-1)
        logits2 = (
            torch.bmm(aux_task_b, proto.permute([0, 2, 1])) / temp
        )
        logits2 = logits2.view(-1, num_proto)

        return logits, logits2
