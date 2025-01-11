# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .meta_model import MetaModel
from core.utils import accuracy
from ..backbone.utils import convert_maml_module
import torch.nn.functional as F

class IFSLUtils(nn.Module):
    def __init__(self, embed_func, feat_dim, ifsl_param, device):
        super(IFSLUtils, self).__init__()
        self.embed_func = embed_func
        self.feat_dim = feat_dim
        self.device = device
        for (key, value) in ifsl_param.items():
            setattr(self, key, value)
        self.linear = nn.Linear(feat_dim, self.class_num)
        self.linear = self._load_state_dict(self.linear, self.cls_path)
        self.softmax = nn.Softmax(dim=1)
        self.features = torch.from_numpy(self.get_pretrain_features()).float().to(self.device)
        if self.normalize_d:
            self.features = self.normalize(self.features)
        self.mean_features = self.features.mean(dim=0)

    def classify(self, x, is_feature=False):
        if is_feature is True:
            return self.softmax(self.linear(x))
        return self.softmax(self.linear(self(x)))

    def _load_state_dict(self, model, state_dict_path):
        if state_dict_path is not None:
            model_state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model_state_dict)
        return model

    def get_pretrain_features(self):
        if self.feature_path is not None:
            return np.load(self.feature_path)
        print("Warning: no pretrain features!")
        return np.zeros((self.class_num, self.feat_dim))

    def normalize(self, x, dim=1):
        x_norm = torch.norm(x, p=2, dim=dim).unsqueeze(dim).expand_as(x).detach()
        x_normalized = x.div(x_norm + 0.00001)
        return x_normalized

    def fuse_proba(self, p1, p2):
        sigmoid = torch.nn.Sigmoid()
        if self.logit_fusion == "linear_sum":
            return p1 + p2
        elif self.logit_fusion == "product":
            return torch.log(sigmoid(p1) * sigmoid(p2))
        elif self.logit_fusion == "sum":
            return torch.log(sigmoid(p1 + p2))
        elif self.logit_fusion == "harmonic":
            p = sigmoid(p1) * sigmoid(p2)
            return torch.log(p / (1 + p))

    def fuse_features(self, x1, x2):
        if self.fusion == "concat":
            return torch.cat((x1, x2), dim=2)
        elif self.fusion == "+":
            return x1 + x2
        elif self.fusion == "-":
            return x1 - x2

    def get_feat_dim(self):
        split_feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == "pd":
            return split_feat_dim + self.num_classes
        else:
            if self.fusion == "concat":
                return split_feat_dim * 2
            else:
                return split_feat_dim

    def fusing(self, support, query):
        support = self.embed_func(support)
        query = self.embed_func(query)
        split_support, support_d = self.get_feature(support)
        split_query, query_d = self.get_feature(query)
        fused_support = self.fuse_features(split_support, support_d)
        fused_query = self.fuse_features(split_query, query_d)
        if self.x_zero:
            c_split_query = torch.zeros_like(split_query).to(self.device)
        else:
            c_split_query = split_support.mean(dim=1).unsqueeze(1).expand(split_query.shape)
        c_fused_query = self.fuse_features(c_split_query, query_d)
        if self.single is True:
            return fused_support, fused_query, c_fused_query
        else:
            return split_support, support_d, split_query, query_d

    def get_split_features(self, x, preprocess=False, center=None, preprocess_method="l2n"):
        # Sequentially cut into n_splits parts
        split_dim = int(self.feat_dim / self.n_splits)
        split_features = torch.zeros(self.n_splits, x.shape[0], split_dim).to(self.device)
        for i in range(self.n_splits):
            start_idx = split_dim * i
            end_idx = split_dim * i + split_dim
            split_features[i] = x[:, start_idx:end_idx]
            if preprocess:
                if preprocess_method != "dl2n":
                    split_features[i] = self.nn_preprocess(split_features[i], center[:, start_idx:end_idx],
                                                           preprocessing=preprocess_method)
                else:
                    if self.normalize_before_center:
                        split_features[i] = self.normalize(split_features[i])
                    centered_data = split_features[i] - center[i]
                    split_features[i] = self.normalize(centered_data)
        return split_features

    def nn_preprocess(self, data, center=None, preprocessing="l2n"):
        if preprocessing == "none":
            return data
        elif preprocessing == "l2n":
            return self.normalize(data)
        elif preprocessing == "cl2n":
            if self.normalize_before_center:
                data = self.normalize(data)
            centered_data = data - center
            return self.normalize(centered_data)

    def calc_pd(self, x):
        with torch.no_grad():
            proba = self.classify(x, True)
        return proba

    def get_d_feature(self, x):
        feat_dim = int(self.feat_dim / self.n_splits)
        if self.d_feature == "ed":
            d_feat_dim = int(self.feat_dim / self.n_splits)
        else:
            d_feat_dim = self.num_classes
        d_feature = torch.zeros(self.n_splits, x.shape[0], d_feat_dim).to(self.device)
        for i in range(self.n_splits):
            start = i * feat_dim
            stop = start + feat_dim
            pd = self.calc_pd(x)
            if self.d_feature == "pd":
                d_feature[i] = pd
            else:
                d_feature[i] = torch.mm(pd, self.features)[:, start:stop]
        return d_feature

    def get_feature(self, x):
        x_d = self.get_d_feature(x)
        if self.normalize_ed:
            x_d = self.normalize(x_d, dim=2)
        x_size = x.shape[0]
        pmean_x = self.mean_features.expand((x_size, self.feat_dim))
        x = self.nn_preprocess(x, pmean_x, preprocessing=self.preprocess_before_split)
        split_x = self.get_split_features(x, preprocess=True, center=pmean_x,
                                          preprocess_method=self.preprocess_after_split)
        return split_x, x_d

    def one_hot(self, y, num_class):      
        return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.softmax = nn.Softmax(dim=1)
        self.c_0 = Variable(torch.zeros(1, feat_dim))
        self.feat_dim = feat_dim

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0, 1)
        K = G.size(0)  # Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r), 1)
            h, c = self.lstmcell(x, (h, c))
            h = h + f
        return h

    def cuda(self):
        super(FullyContextualEmbedding, self).cuda()
        self.c_0 = self.c_0.cuda()
        self.lstmcell = self.lstmcell.cuda()
        return self


class MatchingNetLayer(nn.Module):
    def __init__(self, feat_dim):
        super(MatchingNetLayer, self).__init__()
        self.feat_dim = feat_dim
        self.FCE = FullyContextualEmbedding(self.feat_dim).cuda()
        self.G_encoder = nn.LSTM(self.feat_dim, self.feat_dim, 1, batch_first=True, bidirectional=True).cuda()

    def forward(self, support, query):
        G_encoder = self.G_encoder
        FCE = self.FCE
        out_G = G_encoder(support.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = support + out_G[:, :support.size(1)] + out_G[:, support.size(1):]
        F = FCE(query, G)
        return G, F

    def cuda(self):
        super(MatchingNetLayer, self).cuda()
        self.FCE = self.FCE.cuda()
        self.G_encoder = self.G_encoder.cuda()
        return self


class DMatchingNet(MetaModel):
    def __init__(self, inner_param, feat_dim, ifsl_param, **kwargs):
        super(DMatchingNet, self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.loss_func = nn.NLLLoss()
        self.inner_param = inner_param
        self.ifsl_param = ifsl_param
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        for (key, value) in self.ifsl_param.items():
            setattr(self, key, value)
        self.utils = IFSLUtils(self.emb_func, feat_dim, ifsl_param, self.device)
        assert self.feat_dim % self.n_splits == 0, "feat_dim must be divisible by n_splits"
        if self.use_x_only:
            self.single = False
        if self.single is True:
            self.feat_dim = self.utils.get_feat_dim()
            self.blocks = nn.ModuleList([MatchingNetLayer(self.feat_dim).cuda() for i in range(self.n_splits)])
        else:
            x_feat_dim = int(self.feat_dim / self.n_splits)
            if self.d_feature == "pd":
                d_feat_dim = self.num_classes
            else:
                d_feat_dim = x_feat_dim
            self.x_blocks = nn.ModuleList([MatchingNetLayer(x_feat_dim).cuda() for i in range(self.n_splits)])
            self.d_blocks = nn.ModuleList([MatchingNetLayer(d_feat_dim).cuda() for i in range(self.n_splits)])
        convert_maml_module(self)

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
            episode_support_image = support_image[i].contiguous().reshape(-1, c, h, w)
            episode_query_image = query_image[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            scores = torch.zeros(self.n_splits, episode_query_image.shape[0], episode_support_image.shape[0]).to(self.device)
            c_scores = torch.zeros(self.n_splits, episode_query_image.shape[0], episode_support_image.shape[0]).to(self.device)
            if self.single is True:
                fused_support, fused_query, c_fused_query = self.utils.fusing(episode_support_image,
                                                                              episode_query_image)
                for j in range(self.n_splits):
                    support_new, query_new = self.set_forward_adaptation(self.blocks[j], fused_support[j],
                                                                         fused_query[j])
                    _, c_query_new = self.set_forward_adaptation(self.blocks[j], fused_support[j], c_fused_query[j])
                    scores[j] = self.relu(self.utils.normalize(query_new).mm(
                        self.utils.normalize(support_new).transpose(0, 1))) * self.temp
                    c_scores[j] = self.relu(self.utils.normalize(c_query_new).mm(
                        self.utils.normalize(support_new).transpose(0, 1))) * self.temp
            else:
                split_support, support_d, split_query, query_d = self.utils.fusing(episode_support_image,
                                                                                   episode_query_image)
                for j in range(self.n_splits):
                    support_x_new, query_x_new = self.set_forward_adaptation(self.x_blocks[j], split_support[j],
                                                                             split_query[j])
                    support_d_new, query_d_new = self.set_forward_adaptation(self.d_blocks[j], support_d[j], query_d[j])
                    x_score = self.relu(
                        self.utils.normalize(query_x_new).mm(self.utils.normalize(support_x_new).transpose(0, 1)))
                    d_score = self.relu(
                        self.utils.normalize(query_d_new).mm(self.utils.normalize(support_d_new).transpose(0, 1)))
                    c_x_scores = torch.ones_like(x_score).cuda()
                    if self.use_x_only:
                        scores[j] = x_score * self.temp
                        c_scores[j] = c_x_scores * self.temp
                    else:
                        scores[j] = self.utils.fuse_proba(x_score, d_score) * self.temp
                        c_scores[j] = self.utils.fuse_proba(c_x_scores, d_score) * self.temp
            if self.use_counterfactual:
                scores = scores - c_scores
            scores = self.softmax(scores)
            labels = torch.from_numpy(np.repeat(range(self.way_num),self.shot_num))
            labels = Variable(self.utils.one_hot(labels, self.way_num)).cuda()
            proba = scores.mean(dim=0)
            logprobs = (proba.mm(labels) + 1e-6).log()
            output_list.append(logprobs)
        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
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
            episode_support_target = support_target[i].reshape(-1)
            scores = torch.zeros(self.n_splits, episode_query_image.shape[0], episode_support_image.shape[0]).to(self.device)
            c_scores = torch.zeros(self.n_splits, episode_query_image.shape[0], episode_support_image.shape[0]).to(self.device)
            if self.single is True:
                fused_support, fused_query, c_fused_query = self.utils.fusing(episode_support_image,
                                                                              episode_query_image)
                for j in range(self.n_splits):
                    support_new, query_new = self.set_forward_adaptation(self.blocks[j], fused_support[j],
                                                                         fused_query[j])
                    _, c_query_new = self.set_forward_adaptation(self.blocks[j], fused_support[j], c_fused_query[j])
                    scores[j] = self.relu(self.utils.normalize(query_new).mm(
                        self.utils.normalize(support_new).transpose(0, 1))) * self.temp
                    c_scores[j] = self.relu(self.utils.normalize(c_query_new).mm(
                        self.utils.normalize(support_new).transpose(0, 1))) * self.temp
            else:
                split_support, support_d, split_query, query_d = self.utils.fusing(episode_support_image,
                                                                                   episode_query_image)
                for j in range(self.n_splits):
                    support_x_new, query_x_new = self.set_forward_adaptation(self.x_blocks[j], split_support[j],
                                                                             split_query[j])
                    support_d_new, query_d_new = self.set_forward_adaptation(self.d_blocks[j], support_d[j], query_d[j])
                    x_score = self.relu(
                        self.utils.normalize(query_x_new).mm(self.utils.normalize(support_x_new).transpose(0, 1)))
                    d_score = self.relu(
                        self.utils.normalize(query_d_new).mm(self.utils.normalize(support_d_new).transpose(0, 1)))
                    c_x_scores = torch.ones_like(x_score).cuda()
                    if self.use_x_only:
                        scores[j] = x_score * self.temp
                        c_scores[j] = c_x_scores * self.temp
                    else:
                        scores[j] = self.utils.fuse_proba(x_score, d_score) * self.temp
                        c_scores[j] = self.utils.fuse_proba(c_x_scores, d_score) * self.temp
            if self.use_counterfactual:
                scores = scores - c_scores
            scores = self.softmax(scores)
            labels = torch.from_numpy(np.repeat(range(self.way_num), self.shot_num))
            labels = Variable(self.utils.one_hot(labels,self.way_num)).cuda()
            proba = scores.mean(dim=0)
            logprobs = (proba.mm(labels) + 1e-6).log()
            output_list.append(logprobs)
        output = torch.cat(output_list, dim=0)
        loss = self.loss_func(output, query_target.contiguous().view(-1))
        acc = accuracy(output, query_target.contiguous().view(-1))
        return output, acc, loss

    def set_forward_adaptation(self, model, support, query):
        model.train()
        G, F = model(support, query)
        return G, F
