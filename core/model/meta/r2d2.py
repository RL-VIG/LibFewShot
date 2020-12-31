import torch
from torch import nn

from core.model.abstract_model import AbstractModel
from core.utils import accuracy


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert (A.dim() == 3)
    assert (B.dim() == 3)
    assert (A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1, 2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.

    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).to(b_mat.device)
    b_inv, _ = torch.solve(id_matrix, b_mat)

    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).to(indices.device)
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


class R2D2Head(nn.Module):
    """
    ICLR 2019: Meta-learning with differentiable closed-form solvers
    https://arxiv.org/abs/1805.08136
    """

    def __init__(self, ratio=50.0):
        super(R2D2Head, self).__init__()
        self.ratio = ratio

    def forward(self, query, support, support_labels, n_way, n_shot):
        tasks_per_batch = query.size(0)
        n_support = support.size(1)

        assert (query.dim() == 3)
        assert (support.dim() == 3)
        assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
        assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

        id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).to(query.device)

        # Compute the dual form solution of the ridge regression.
        # W = X^T(X X^T - lambda * I)^(-1) Y
        ridge_sol = computeGramMatrix(support, support) + self.ratio * id_matrix
        ridge_sol = binv(ridge_sol)
        ridge_sol = torch.bmm(support.transpose(1, 2), ridge_sol)
        ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)

        # Compute the classification score.
        # score = W X
        logits = torch.bmm(query, ridge_sol)

        return logits


class R2D2(AbstractModel):
    def __init__(self, way_num, shot_num, query_num, feature, device):
        super(R2D2, self).__init__(way_num, shot_num, query_num, feature, device)
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = R2D2Head()
        self._init_network()

    def set_forward(self, batch, ):
        query_images, query_targets, support_images, support_targets = batch
        query_images = torch.cat(query_images, 0)
        query_targets = torch.cat(query_targets, 0)
        support_images = torch.cat(support_images, 0)
        support_targets = torch.cat(support_targets, 0)
        query_images = query_images.to(self.device)
        query_targets = query_targets.to(self.device)
        support_images = support_images.to(self.device)
        support_targets = support_targets.to(self.device)

        with torch.no_grad():
            support_feat = self.model_func(support_images)
        classifier_copy = self.train_loop(support_feat, support_targets)

        query_feat = self.model_func(query_images)
        output = classifier_copy(query_feat)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1

    def set_forward_loss(self, batch, ):
        query_images, query_targets, support_images, support_targets = batch
        query_images = torch.cat(query_images, 0)
        query_targets = torch.cat(query_targets, 0)
        support_images = torch.cat(support_images, 0)
        support_targets = torch.cat(support_targets, 0)
        query_images = query_images.to(self.device)
        query_targets = query_targets.to(self.device)
        support_images = support_images.to(self.device)
        support_targets = support_targets.to(self.device)

        emb_query = self.model_func(query_images)
        emb_support = self.model_func(support_images)

        # TODO 第1维是episode_num，暂时默认为1
        emb_query = emb_query.unsqueeze(0)
        emb_support = emb_support.unsqueeze(0)

        output = self.classifier(emb_query, emb_support, support_targets, self.way_num, self.shot_num)

        output = output.squeeze()
        loss = self.loss_func(output, query_targets)
        prec1, _ = accuracy(output, query_targets, topk=(1, 3))

        return output, prec1, loss

    def train_loop(self, support_feat, support_targets):
        raise NotImplementedError

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError
