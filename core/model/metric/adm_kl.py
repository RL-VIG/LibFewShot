import torch
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel
# https://github.com/WenbinLee/ADM

class KL_Layer(nn.Module):
    def __init__(self,train_way, train_shot, train_query,n_k,device,CMS = False):
        super(KL_Layer, self).__init__()
        self.train_way = train_way
        self.train_shot = train_shot
        self.train_query = train_query
        self.n_k= n_k
        self.device = device
        self.CMS = CMS

    def _cal_cov_matrix_batch(self, feat):  # feature: e *  Batch * descriptor_num * 64
        e, _, n_local, c = feat.size()
        feature_mean = torch.mean(feat, 2, True)  # e * Batch * 1 * 64
        feat = feat - feature_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat) #  ebc1 * eb1c = ebcc
        cov_matrix = torch.div(cov_matrix, n_local - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(self.device) # broadcast from the last dim

        return feature_mean, cov_matrix

    def _cal_cov_batch(self, feat):  # feature: e * 25 * 64 * 21 * 21
        e, b, c, h, w = feat.size()
        feat = feat.view(e, b, c, -1).permute(0, 1, 3, 2)
        feat_mean = torch.mean(feat, 2, True)  # e * Batch * 1 * 64
        feat = feat - feat_mean
        cov_matrix = torch.matmul(feat.permute(0, 1, 3, 2), feat)
        cov_matrix = torch.div(cov_matrix, h * w - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(self.device)

        return feat_mean, cov_matrix

    def _calc_kl_dist_batch(self, mean1, cov1, mean2, cov2):
        """

        :param mean1: e * 75 * 1 * 64
        :param cov1: e * 75 * 64 * 64
        :param mean2: e * 5 * 1 * 64
        :param cov2: e * 5 * 64 * 64
        :return:
        """

        cov2_inverse = torch.inverse(cov2)  # e * 5 * 64 * 64
        mean_diff = -(mean1 - mean2.squeeze(2).unsqueeze(1))  # e * 75 * 5 * 64

        # Calculate the trace
        matrix_prod = torch.matmul(cov1.unsqueeze(2), cov2_inverse.unsqueeze(1))  # e * 75 * 5 * 64 * 64
        # trace_dist = [[torch.trace(matrix_prod[e][j][i]).unsqueeze(0) # modified for multi-task
        #                for j in range(matrix_prod.size(1))
        #                for i in range(matrix_prod.size(2))]
        #               for e in range(matrix_prod.size(0))] # list of trace_dist
        # trace_dist = torch.stack([torch.cat(trace_dist_list, 0) for trace_dist_list in trace_dist]) #
        # trace_dist = trace_dist.view(matrix_prod.size(0),matrix_prod.size(1), matrix_prod.size(2))  # e * 75 * 5
        trace_dist = torch.diagonal(matrix_prod,offset=0,dim1=-2,dim2=-1)  # e * 75 * 5 * 64
        trace_dist = torch.sum(trace_dist,dim=-1) # e * 75 * 5

        # Calcualte the Mahalanobis Distance
        maha_prod = torch.matmul(mean_diff.unsqueeze(3), cov2_inverse.unsqueeze(1))  # e * 75 * 5 * 1 * 64
        maha_prod = torch.matmul(maha_prod, mean_diff.unsqueeze(4))  # e * 75 * 5 * 1 * 1
        maha_prod = maha_prod.squeeze(4)
        maha_prod = maha_prod.squeeze(3)  # e * 75 * 5

        matrix_det = torch.logdet(cov2).unsqueeze(1) - torch.logdet(cov1).unsqueeze(2)

        kl_dist = trace_dist + maha_prod + matrix_det - mean1.size(3)

        return kl_dist / 2.

    def _cal_support_remaining(self, S):   # S: e * 5 * 441 * 64
        e,w,d,c = S.shape
        episode_indices = torch.tensor([j for i in range(
                S.size(1)) for j in range(S.size(1)) if i != j]).to(self.device)
        S_new = torch.index_select(S, 1, episode_indices)
        S_new = S_new.view([e,w,-1,c])

        # S_new = [] # 5 * 441 * 64 ADM source code
        # for ii in range(S.size(0)):
        #
        #     indices = [j for j in range(S.size(0))]
        #     indices.pop(ii)
        #     indices = torch.tensor(indices).cuda()
        #
        #     S_clone = S.clone()
        #     S_remain = torch.index_select(S_clone, 0, indices)           # 4 * 441 * 64
        #     S_remain = S_remain.contiguous().view(-1, S_remain.size(2))  # 1764 * 64
        #     S_new.append(S_remain.unsqueeze(0))
        #
        # S_new = torch.cat(S_new, 0)   # 5 * 1764 * 64
        
        return S_new

    # Calculate KL divergence Distance
    def _cal_adm_sim(self, query_feat, support_feat):
        """

        :param query_feat: e * 75 * 64 * 21 * 21
        :param support_feat: e * 25 * 64 * 21 * 21
        :return:
        """
        # query_mean: e * 75 * 1 * 64  query_cov: e * 75 * 64 * 64
        e, b, c, h, w = query_feat.size()
        e, s, _, _, _ = support_feat.size()
        query_mean, query_cov = self._cal_cov_batch(query_feat)

        query_feat = query_feat.view(e, b, c, -1).permute(0, 1, 3, 2).contiguous()

        # Calculate the mean and covariance of the support set
        support_feat = support_feat.view(e, s, c, -1).permute(0, 1, 3, 2).contiguous()
        support_set = support_feat.view(e, self.train_way, self.train_shot * h * w, c)

        # s_mean: e * 5 * 1 * 64  s_cov: e * 5 * 64 * 64
        s_mean, s_cov = self._cal_cov_matrix_batch(support_set)

        # Calculate the Wasserstein Distance
        kl_dis = -self._calc_kl_dist_batch(query_mean, query_cov, s_mean, s_cov)  # e * 75 * 5

        if self.CMS: # ADM_KL_CMS
            # Find the remaining support set
            support_set_remain = self._cal_support_remaining(support_set)
            s_remain_mean, s_remain_cov = self._cal_cov_matrix_batch(support_set_remain) # s_remain_mean: e * 5 * 1 * 64  s_remain_cov: e * 5 * 64 * 64
            kl_dis2 = self._calc_kl_dist_batch(query_mean, query_cov, s_remain_mean, s_remain_cov)  # e * 75 * 5
            kl_dis = kl_dis+kl_dis2

        return kl_dis

    def forward(self, query_feat, support_feat):
        return self._cal_adm_sim(query_feat,support_feat)


class ADM_KL(MetricModel):
    def __init__(self, train_way, train_shot, train_query, emb_func, device, n_k=3, CMS=False):
        super(ADM_KL, self).__init__(train_way, train_shot, train_query, emb_func, device)
        self.n_k = n_k
        self.kl_layer = KL_Layer(train_way, train_shot, train_query, n_k, device, CMS)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.train_way * (self.train_shot + self.train_query))
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=2)

        output = self.kl_layer(query_feat, support_feat).view(episode_size * self.train_way * self.train_query, -1)
        acc = accuracy(output, query_target)
        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (self.train_way * (self.train_shot + self.train_query))
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=2)
        # assume here we will get n_dim=5
        output = self.kl_layer(query_feat, support_feat).view(episode_size * self.train_way * self.train_query, -1)
        loss = self.loss_func(output, query_target)
        acc = accuracy(output, query_target)
        return output, acc, loss
