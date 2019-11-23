import abc
import numpy as np
import torch

class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        marginalLoss = torch.zeros_like(x_scores)

        # TODO: refactor without explicit loops
        example_idx = 0
        for classification in y:
            marginalLoss[example_idx] = x_scores[example_idx] - x_scores[example_idx][classification] + self.delta
            marginalLoss[example_idx][classification] -= self.delta
            # print("example[", example_idx, "] scores: ", x_scores[example_idx])
            # print("example[", example_idx, "] real class score: ", x_scores[example_idx][classification])
            # print("example[", example_idx, "] marginal loss: ", marginalLoss[example_idx])
            example_idx += 1

        zeros_mat = torch.zeros_like(marginalLoss)
        loss = torch.zeros([1,])

        L_i = torch.max(marginalLoss, zeros_mat)

        # TODO: refactor without explicit loops
        # for classification in L_i:
            # loss += torch.sum(classification)
            # loss -= self.delta
        loss = torch.sum(L_i)
        
        loss /= x_scores.shape[0]
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['marginalLoss'] = marginalLoss
        self.grad_ctx['X_rows_len'] = x.shape[0]
        self.grad_ctx['X_cols_len'] = x.shape[1]
        self.grad_ctx['X_scores_cols'] = x_scores.shape[1]
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        M = self.grad_ctx['marginalLoss']
        N = self.grad_ctx['X_rows_len']
        C = self.grad_ctx['X_scores_cols']
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']
        G = torch.zeros([N, C])

        G = M

        # j != y_i => grad = x_i
        G[M > 0] = 1
        row_sum = torch.sum(G, axis=1)
        tensorlist = torch.LongTensor(list(range(N)))

        # j == y_i => grad = -x_i * sum[m_i,j > 0]
        G[tensorlist, y] = (-1) * row_sum.t()

        grad = torch.mm(x.t(), G)

        grad /= N
        # ========================

        return grad
