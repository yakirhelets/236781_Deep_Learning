import abc
import numpy as np

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
        print(x.shape) # N*D
        print(x_scores.shape) # N*C
        print(y.shape) # N row vector
        print(y_predicted.shape) # N row vector

        diff = x_scores
        print(diff.shape)

        # Naive approach - explicit loop:
        example_idx = 0
        for classification in y:
            # print(x_scores[example_idx], y[classification], x_scores[example_idx][y[classification]])
            diff[example_idx] = x_scores[example_idx] - x_scores[example_idx][y[classification]] + self.delta
            example_idx += 1

        loss = torch.zeros([1,])

        zeros_mat = torch.zeros_like(diff)
        print(diff)

        diff_2 = torch.max(diff, zeros_mat)
        print(diff_2)
        # loss = torch.sum(diff_2)
        for classification in diff_2:
            loss += torch.sum(classification)
            loss -= self.delta
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx = diff
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
        M = self.grad_ctx['M_matrix']
        N = self.grad_ctx['X_rows_len']
        D = self.grad_ctx['X_cols_len']
        C = self.grad_ctx['y_size']
        x = self.grad_ctx['x']
        # G = TODO G should be of shape N x C
        new_mat = None
        for j in range(C):
            col = None
            for i in range(N):
                if (M[i,j] > 0):
                    next_element = x[i, :]
                else:
                    next_element = np.zeros((1, D))
                col += next_element
            col /= N
            # TODO add col to new_mat

        grad = np.matmul(x.transpose(), G)
        # ========================

        return grad
