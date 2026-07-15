# -*- coding: utf-8 -*-
"""
LDP-Net: Revisiting Prototypical Network for Cross Domain Few-Shot Learning.

This implementation follows LibFewShot's episodic metric-model interface while
adapting the original LDP-Net training and testing ideas.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class LDPNet(MetricModel):
    def __init__(
        self,
        lamba1=1.0,
        lamba2=0.15,
        momentum=0.998,
        beta=0.5,
        query_aug_times=1,
        use_lr_classifier=True,
        transductive=False,
        transductive_iter=7,
        transductive_k=10,
        lr_max_iter=1000,
        local_query_crops=0,
        eps=1e-6,
        **kwargs
    ):
        super(LDPNet, self).__init__(**kwargs)
        self.lamba1 = lamba1
        self.lamba2 = lamba2
        self.momentum = momentum
        self.beta = beta
        self.query_aug_times = max(int(query_aug_times), 1)
        self.use_lr_classifier = use_lr_classifier
        self.transductive = transductive
        self.transductive_iter = transductive_iter
        self.transductive_k = transductive_k
        self.lr_max_iter = lr_max_iter
        self.local_query_crops = local_query_crops
        self.eps = eps

        self.loss_func = nn.CrossEntropyLoss()
        self.ema_emb_func = copy.deepcopy(self.emb_func)
        for param in self.ema_emb_func.parameters():
            param.requires_grad = False

        if self.query_aug_times > 1 and self.query_num % self.query_aug_times != 0:
            raise ValueError(
                "query_num must be divisible by query_aug_times in LDPNet. "
                "Set classifier.kwargs.query_aug_times to match augment_times_query."
            )

    def train(self, mode=True):
        super(LDPNet, self).train(mode)
        self.ema_emb_func.train(mode)
        return self

    @torch.no_grad()
    def _momentum_update_ema(self):
        for param, ema_param in zip(
            self.emb_func.parameters(), self.ema_emb_func.parameters()
        ):
            ema_param.data.mul_(self.momentum).add_(
                param.data, alpha=1.0 - self.momentum
            )

    def _flatten_feat(self, feat):
        if feat.dim() > 2:
            feat = F.adaptive_avg_pool2d(feat, 1)
            feat = torch.flatten(feat, 1)
        return feat

    def _power_transform(self, feat):
        if self.beta is None:
            return feat
        return torch.pow(feat.clamp_min(0.0), self.beta)

    def _proto_logits(self, support_feat, query_feat):
        episode_size, _, feat_dim = support_feat.size()
        support_feat = support_feat.view(
            episode_size, self.way_num, self.shot_num, feat_dim
        )
        proto_feat = torch.mean(support_feat, dim=2)
        return -torch.sum(
            torch.pow(query_feat.unsqueeze(2) - proto_feat.unsqueeze(1), 2),
            dim=3,
        )

    def _anchor_query(self, query_feat):
        if self.query_aug_times <= 1:
            return query_feat
        episode_size, _, feat_dim = query_feat.size()
        query_per_aug = self.query_num // self.query_aug_times
        query_feat = query_feat.view(
            episode_size, self.way_num, self.query_aug_times, query_per_aug, feat_dim
        )
        query_feat = query_feat.permute(0, 2, 1, 3, 4).contiguous()
        return query_feat[:, 0].view(episode_size, self.way_num * query_per_aug, feat_dim)

    def _aug_query(self, query_feat):
        if self.query_aug_times <= 1:
            return None
        episode_size, _, feat_dim = query_feat.size()
        query_per_aug = self.query_num // self.query_aug_times
        query_feat = query_feat.view(
            episode_size, self.way_num, self.query_aug_times, query_per_aug, feat_dim
        )
        query_feat = query_feat.permute(0, 2, 1, 3, 4).contiguous()
        return query_feat[:, 1:].view(
            episode_size,
            (self.query_aug_times - 1) * self.way_num * query_per_aug,
            feat_dim,
        )

    def _anchor_target(self, query_target):
        if self.query_aug_times <= 1:
            return query_target
        episode_size = query_target.size(0)
        query_per_aug = self.query_num // self.query_aug_times
        query_target = query_target.view(
            episode_size, self.way_num, self.query_aug_times, query_per_aug
        )
        query_target = query_target.permute(0, 2, 1, 3).contiguous()
        return query_target[:, 0].view(episode_size, self.way_num * query_per_aug)

    def _consistency_loss(self, anchor_output, aug_output, aug_num=None):
        if aug_output is None:
            return anchor_output.new_tensor(0.0), anchor_output.new_tensor(0.0)

        episode_size, anchor_num, way_num = anchor_output.size()
        aug_num = self.query_aug_times - 1 if aug_num is None else aug_num
        query_per_aug = anchor_num // self.way_num

        anchor_prob = F.softmax(anchor_output, dim=-1)
        aug_prob = F.softmax(aug_output, dim=-1)

        anchor_repeat = anchor_prob.unsqueeze(1).repeat(1, aug_num, 1, 1)
        anchor_repeat = anchor_repeat.view(episode_size, aug_num * anchor_num, way_num)
        self_image_loss = -torch.mean(
            torch.sum(anchor_repeat * torch.log(aug_prob.clamp_min(self.eps)), dim=-1)
        )

        anchor_by_class = anchor_prob.view(
            episode_size, self.way_num, query_per_aug, way_num
        )
        aug_by_class = aug_prob.view(
            episode_size, aug_num, self.way_num, query_per_aug, way_num
        )
        # Match the author's implementation: the two query indices are sampled
        # from NumPy's RNG.  Using the CUDA RNG here changes the fixed-seed
        # training trajectory even when every other paper setting is identical.
        global_idx = [
            np.random.permutation(query_per_aug)[0] for _ in range(episode_size)
        ]
        local_idx = [
            np.random.permutation(query_per_aug)[0] for _ in range(episode_size)
        ]

        global_prob = []
        local_prob = []
        for epi in range(episode_size):
            global_prob.append(anchor_by_class[epi, :, global_idx[epi], :])
            local_prob.append(aug_by_class[epi, :, :, local_idx[epi], :])
        global_prob = torch.stack(global_prob, dim=0)
        local_prob = torch.stack(local_prob, dim=0)
        global_prob = global_prob.unsqueeze(1).expand(-1, aug_num, -1, -1)
        cross_image_loss = -torch.mean(
            torch.sum(global_prob * torch.log(local_prob.clamp_min(self.eps)), dim=-1)
        )
        return self_image_loss, cross_image_loss

    def set_forward_loss(self, batch):
        images, global_targets = batch[:2]
        explicit_query_aug_images = batch[2] if len(batch) > 2 else None
        images = images.to(self.device)

        feat = self._flatten_feat(self.emb_func(images))
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        anchor_query_feat = self._anchor_query(query_feat)
        anchor_query_target = self._anchor_target(query_target)

        anchor_output = self._proto_logits(support_feat, anchor_query_feat)
        loss = self.loss_func(
            anchor_output.reshape(-1, self.way_num), anchor_query_target.reshape(-1)
        )

        if explicit_query_aug_images is not None:
            explicit_query_aug_images = explicit_query_aug_images.to(self.device)
            local_crops, query_count, c, h, w = explicit_query_aug_images.size()
            with torch.no_grad():
                aug_feat = self._flatten_feat(
                    self.ema_emb_func(
                        explicit_query_aug_images.view(local_crops * query_count, c, h, w)
                    )
                )
            aug_feat = aug_feat.view(
                local_crops,
                -1,
                self.way_num * (self.query_num // self.query_aug_times),
                aug_feat.size(-1),
            )
            aug_feat = aug_feat.permute(1, 0, 2, 3).contiguous()
            aug_query_feat = aug_feat.view(
                aug_feat.size(0), local_crops * aug_feat.size(2), aug_feat.size(3)
            )
            aug_output = self._proto_logits(support_feat, aug_query_feat)
            self_image_loss, cross_image_loss = self._consistency_loss(
                anchor_output, aug_output, aug_num=local_crops
            )
            loss = loss + self.lamba1 * self_image_loss + self.lamba2 * cross_image_loss
        elif self.query_aug_times > 1:
            with torch.no_grad():
                ema_feat = self._flatten_feat(self.ema_emb_func(images))
                _, ema_query_feat, _, _ = self.split_by_episode(ema_feat, mode=1)
            aug_query_feat = self._aug_query(ema_query_feat)
            aug_output = self._proto_logits(support_feat, aug_query_feat)
            self_image_loss, cross_image_loss = self._consistency_loss(
                anchor_output, aug_output
            )
            loss = loss + self.lamba1 * self_image_loss + self.lamba2 * cross_image_loss

        output = anchor_output.reshape(-1, self.way_num)
        acc = accuracy(output, anchor_query_target.reshape(-1))
        return output, acc, loss

    def set_forward(self, batch):
        images, global_targets = batch
        images = images.to(self.device)
        feat = self._power_transform(self._flatten_feat(self.emb_func(images)))
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        query_feat = self._anchor_query(query_feat)
        query_target = self._anchor_target(query_target)

        if self.use_lr_classifier:
            output = self._logistic_regression_forward(
                support_feat, support_target, query_feat
            )
        else:
            output = self._proto_logits(support_feat, query_feat).reshape(
                -1, self.way_num
            )
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def _logistic_regression_forward(self, support_feat, support_target, query_feat):
        from sklearn.linear_model import LogisticRegression

        output_list = []
        episode_size = support_feat.size(0)
        for epi in range(episode_size):
            support_np = support_feat[epi].detach().cpu().numpy()
            query_np = query_feat[epi].detach().cpu().numpy()
            target_np = support_target[epi].detach().cpu().numpy()

            classifier = LogisticRegression(max_iter=self.lr_max_iter)
            classifier.fit(support_np, target_np)
            prob = classifier.predict_proba(query_np)

            if self.transductive:
                prob = self._transductive_refine(
                    classifier,
                    support_feat[epi],
                    support_target[epi],
                    query_feat[epi],
                    prob,
                )

            output_list.append(torch.from_numpy(prob).to(self.device).float())
        return torch.cat(output_list, dim=0)

    def _transductive_refine(
        self, classifier, support_feat, support_target, query_feat, prob
    ):
        from sklearn.linear_model import LogisticRegression

        feat_dim = support_feat.size(-1)
        support_by_class = support_feat.view(self.way_num, self.shot_num, feat_dim)
        y_proto = np.arange(self.way_num)

        for _ in range(self.transductive_iter):
            # The reference Tr() applies softmax to sklearn's predict_proba
            # output and keeps its float64 precision for confidence ranking.
            prob_tensor = torch.from_numpy(prob).to(self.device)
            prob_tensor = F.softmax(prob_tensor, dim=1)
            score, pred = prob_tensor.max(dim=1)
            refined_feat = []
            refined_target = []
            for cls_idx in range(self.way_num):
                selected = torch.nonzero(pred == cls_idx, as_tuple=False).view(-1)
                if selected.numel() > self.transductive_k:
                    selected_score = score[selected]
                    _, top_idx = torch.topk(selected_score, self.transductive_k)
                    selected = selected[top_idx]

                if self.shot_num == 1:
                    class_feat = support_by_class[cls_idx]
                    if selected.numel() > 0:
                        class_feat = torch.cat([class_feat, query_feat[selected]], dim=0)
                    refined_feat.append(class_feat.mean(dim=0, keepdim=True))
                else:
                    class_feat = support_by_class[cls_idx]
                    if selected.numel() > 0:
                        class_feat = torch.cat([class_feat, query_feat[selected]], dim=0)
                    refined_feat.append(class_feat)
                    refined_target.extend([cls_idx] * class_feat.size(0))

            if self.shot_num == 1:
                train_x = torch.cat(refined_feat, dim=0).detach().cpu().numpy()
                train_y = y_proto
            else:
                train_x = torch.cat(refined_feat, dim=0).detach().cpu().numpy()
                train_y = np.asarray(refined_target)

            classifier = LogisticRegression(max_iter=self.lr_max_iter)
            classifier.fit(train_x, train_y)
            prob = classifier.predict_proba(query_feat.detach().cpu().numpy())
        return prob
