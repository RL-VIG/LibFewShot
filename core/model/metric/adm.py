import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class ADM(MetricModel):
    def __init__(self, way_num, shot_num, query_num, model_func, device, n_k=3):
        super(ADM, self).__init__(way_num, shot_num, query_num, model_func, device)
        self.n_k = n_k

        self.norm_layer = nn.BatchNorm1d(self.way_num * 2, affine=True)
        self.fc_layer = nn.Conv1d(1, 1, kernel_size=2, stride=1, dilation=5, bias=False)

        self.loss_func = nn.CrossEntropyLoss()
        self._init_network()

    def set_forward(self, batch, ):
        """

        :param batch:
        :return:
        """
        support_images, support_targets, query_images, query_targets = \
            self.progress_batch(batch)

        query_feat = self.model_func(query_images)
        support_feat = self.model_func(support_images)

        output = self._cal_adm_sim(query_feat, support_feat)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))
        return output, prec1

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        support_images, support_targets, query_images, query_targets = \
            self.progress_batch(batch)

        query_feat = self.model_func(query_images)
        support_feat = self.model_func(support_images)

        output = self._cal_adm_sim(query_feat, support_feat)
        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))
        return output, prec1, loss

    def _cal_cov_matrix_batch(self, feat):  # feature: Batch * descriptor_num * 64
        _, n_local, c = feat.size()
        feature_mean = torch.mean(feat, 1, True)  # Batch * 1 * 64
        feat = feat - feature_mean
        cov_matrix = torch.matmul(feat.permute(0, 2, 1), feat)
        cov_matrix = torch.div(cov_matrix, n_local - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(self.device)

        return feature_mean, cov_matrix

    def _cal_cov_batch(self, feat):  # feature: 25 * 64 * 21 * 21
        b, c, h, w = feat.size()
        feat = feat.view(b, c, -1).permute(0, 2, 1)
        feat_mean = torch.mean(feat, 1, True)  # Batch * 1 * 64
        feat = feat - feat_mean
        cov_matrix = torch.matmul(feat.permute(0, 2, 1), feat)
        cov_matrix = torch.div(cov_matrix, h * w - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(c).to(self.device)

        return feat_mean, cov_matrix

    def _calc_kl_dist_batch(self, mean1, cov1, mean2, cov2):
        """

        :param mean1: 75 * 1 * 64
        :param cov1: 75 * 64 * 64
        :param mean2: 5 * 1 * 64
        :param cov2: 5 * 64 * 64
        :return:
        """

        cov2_inverse = torch.inverse(cov2)  # 5 * 64 * 64
        mean_diff = -(mean1 - mean2.squeeze(1))  # 75 * 5 * 64

        # Calculate the trace
        matrix_prod = torch.matmul(cov1.unsqueeze(1), cov2_inverse)  # 75 * 5 * 64 * 64
        trace_dist = [torch.trace(matrix_prod[j][i]).unsqueeze(0)
                      for j in range(matrix_prod.size(0))
                      for i in range(matrix_prod.size(1))]
        trace_dist = torch.cat(trace_dist, 0)
        trace_dist = trace_dist.view(matrix_prod.size(0), matrix_prod.size(1))  # 75 * 5

        # Calcualte the Mahalanobis Distance
        maha_prod = torch.matmul(mean_diff.unsqueeze(2), cov2_inverse)  # 75 * 5 * 1 * 64
        maha_prod = torch.matmul(maha_prod, mean_diff.unsqueeze(3))  # 75 * 5 * 1 * 1
        maha_prod = maha_prod.squeeze(3)
        maha_prod = maha_prod.squeeze(2)  # 75 * 5

        matrix_det = torch.logdet(cov2) - torch.logdet(cov1).unsqueeze(1)
        kl_dist = trace_dist + maha_prod + matrix_det - mean1.size(2)

        return kl_dist / 2.

    # Calculate KL divergence Distance
    def _cal_adm_sim(self, query_feat, support_feat):
        """

        :param query_feat: 75 * 64 * 21 * 21
        :param support_feat: 25 * 64 * 21 * 21
        :return:
        """
        # query_mean: 75 * 1 * 64  query_cov: 75 * 64 * 64
        b, c, h, w = query_feat.size()
        s, _, _, _ = support_feat.size()
        query_mean, query_cov = self._cal_cov_batch(query_feat)

        query_feat = query_feat.view(b, c, -1).permute(0, 2, 1).contiguous()

        # Calculate the mean and covariance of the support set
        support_feat = support_feat.view(s, c, -1).permute(0, 2, 1).contiguous()
        support_set = support_feat.view(self.way_num, self.shot_num * h * w, c)
        # s_mean: 5 * 1 * 64  s_cov: 5 * 64 * 64
        s_mean, s_cov = self._cal_cov_matrix_batch(support_set)

        # Calculate the Wasserstein Distance
        kl_dis = -self._calc_kl_dist_batch(query_mean, query_cov, s_mean, s_cov)  # 75 * 5

        # Calculate the Image-to-Class Similarity
        query_norm = F.normalize(query_feat, p=2, dim=2)
        support_norm = F.normalize(support_feat, p=2, dim=2)
        support_norm = support_norm.view(self.way_num, self.shot_num * h * w, c)

        # cosine similarity between a query set and a support set
        # 75 * 5 * 441 * 2205
        inner_prod_matrix = torch.matmul(query_norm.unsqueeze(1),
                                         support_norm.permute(0, 2, 1))

        # choose the top-k nearest neighbors
        # 75 * 5 * 441 * 1
        topk_value, topk_index = torch.topk(inner_prod_matrix, self.n_k, 3)
        inner_sim = torch.sum(torch.sum(topk_value, 3), 2)  # 75 * 5

        # Using FC layer to combine two parts ---- The original
        adm_sim_soft = torch.cat((kl_dis, inner_sim), 1)
        adm_sim_soft = self.norm_layer(adm_sim_soft).unsqueeze(1)
        adm_sim_soft = self.fc_layer(adm_sim_soft).squeeze(1)

        return adm_sim_soft
