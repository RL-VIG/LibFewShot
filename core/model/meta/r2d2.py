import torch
from torch import nn

from core.utils import accuracy
from .meta_model import MetaModel


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

    def __init__(self):
        super(R2D2Head, self).__init__()

    def forward(self, query, support, support_labels, n_way, n_shot, ratio=50.0):
        tasks_per_batch = query.size(0)
        n_support = support.size(1)

        # assert (query.dim() == 3)
        # assert (support.dim() == 3)
        # assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
        # assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

        support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
        support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

        id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).to(query.device)

        # Compute the dual form solution of the ridge regression.
        # W = X^T(X X^T - lambda * I)^(-1) Y
        ridge_sol = computeGramMatrix(support, support) + ratio * id_matrix
        ridge_sol = binv(ridge_sol)
        ridge_sol = torch.bmm(support.transpose(1, 2), ridge_sol)
        ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)

        # Compute the classification score.
        # score = W X
        logits = torch.bmm(query, ridge_sol)

        return logits


class R2D2(MetaModel):
    def __init__(self, way_num, shot_num, query_num, feature, device, inner_optim=None, inner_train_iter=10):
        super(R2D2, self).__init__(way_num, shot_num, query_num, feature, device)
        self.register_parameter('alpha', nn.Parameter(torch.tensor([1.])))
        self.register_parameter('beta', nn.Parameter(torch.tensor([0.])))
        self.register_parameter('gamma', nn.Parameter(torch.tensor([50.])))
        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = R2D2Head()
        self.inner_optim = inner_optim
        self.inner_train_iter = inner_train_iter
        self._init_network()

    def set_forward(self, batch, ):
        support_images, support_targets, query_images, query_targets = \
            self.progress_batch2(batch)
        episode = support_images.size(0)

        prec1_list = []
        output_list = []
        for i in range(episode):
            episode_support_images = support_images[i, :]
            episode_query_images = query_images[i, :]
            episode_support_targets = support_targets[i, :].contiguous().view(-1)
            episode_query_targets = query_targets[i, :].contiguous().view(-1)

            emb_support = self.model_func(episode_support_images)
            emb_query = self.model_func(episode_query_images)

            emb_support = emb_support.unsqueeze(0)
            emb_query = emb_query.unsqueeze(0)

            output = self.classifier(emb_query, emb_support, episode_support_targets, self.way_num, self.shot_num,
                                     self.gamma)
            output = self.alpha * output + self.beta
            prec1, _ = accuracy(output.squeeze(), episode_query_targets, topk=(1, 3))

            prec1_list.append(prec1)

        prec1 = torch.mean(torch.tensor(prec1_list))
        output = torch.tensor(output_list)
        return output, prec1

    def set_forward_loss(self, batch, ):
        support_images, support_targets, query_images, query_targets = \
            self.progress_batch2(batch)
        episode = support_images.size(0)
        # emb_feat = self.model_func(images)

        loss_list = []
        prec1_list = []
        output_list = []
        for i in range(episode):
            episode_support_images = support_images[i, :]
            episode_query_images = query_images[i, :]
            episode_support_targets = support_targets[i, :].contiguous().view(-1)
            episode_query_targets = query_targets[i, :].contiguous().view(-1)

            emb_support = self.model_func(episode_support_images)
            emb_query = self.model_func(episode_query_images)

            emb_support = emb_support.unsqueeze(0)
            emb_query = emb_query.unsqueeze(0)

            output = self.classifier(emb_query, emb_support, episode_support_targets, self.way_num, self.shot_num,
                                     self.gamma)
            output = self.alpha * output + self.beta
            loss = self.loss_func(output.squeeze(), episode_query_targets)
            prec1, _ = accuracy(output.squeeze(), episode_query_targets, topk=(1, 3))

            loss_list.append(loss)
            prec1_list.append(prec1)

        loss = torch.mean(torch.stack(loss_list))
        prec1 = torch.mean(torch.tensor(prec1_list))
        output = torch.tensor(output_list)
        return output, prec1, loss

    def train_loop(self, emb_support, support_targets):
        raise NotImplementedError

    def test_loop(self, *args, **kwargs):
        raise NotImplementedError

    def set_forward_adaptation(self, *args, **kwargs):
        raise NotImplementedError
