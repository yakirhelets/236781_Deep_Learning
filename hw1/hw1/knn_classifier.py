import numpy as np
import torch
from scipy import stats
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from . import dataloaders

class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        x_train = torch.Tensor()
        y_train = torch.Tensor()

        for _, (samples, classes) in enumerate(dl_train):
            x_train = torch.cat([x_train, samples], dim=0)
            y_train = torch.cat([y_train, classes.float()], dim=0)

        n_classes = (np.unique(y_train.round().numpy())).size
        # ========================

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            # Get the ith column from the dist matrix
            ith_col = dist_matrix[:, i]

            # Pick the smallest k values from the column
            values, indices = torch.topk(ith_col, self.k, dim=0, largest=False, sorted=False, out=None)
            
            # Bring label of k closest from y_train
            labels = [int(self.y_train[j].item()) for j in indices]
            labels_arr = np.array(labels)
            
            # Get the most common label
            most_common_label = stats.mode(labels_arr)
           
            # Set it as the prediction
            y_pred[i] = torch.as_tensor(np.array([most_common_label[0][0]]))
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.

    dists = None
    # ====== YOUR CODE: ======
    x1_squared = (x1**2).sum(1).view(-1, 1)
    x2_squared = (x2**2).sum(1).view(1, -1)

    dists = torch.sqrt(x1_squared + x2_squared - 2*x1.mm(x2.t()))
    # ========================
    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    equal_elements = 0
    equal_elements += torch.sum(y == y_pred).type(torch.float32)
    accuracy = equal_elements / len(y)
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        # Split to num_folds:
        lengths = [int(round(len(ds_train)/num_folds)) for _ in range(num_folds)]
        subsets = torch.utils.data.random_split(ds_train, lengths)

        for j in range(num_folds):
            accuracy_list = []

            # Test set
            test_set = subsets[j]

            # Train set
            train_subsets = [subset for subset in subsets if subset != subsets[j]]
            train_set = torch.utils.data.ConcatDataset(train_subsets)
            dl_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, num_workers=2)
            
            model.train(dl_train)

            acc = 0
            for example in test_set:
                # example[0] is a tensor
                y_pred = model.predict(example[0].reshape(1,-1))
                y_true = torch.tensor([example[1]])

                acc += accuracy(y_pred, y_true).item()

            # Accuracy for each fold
            fold_accuracy = acc/len(ds_train)
            accuracy_list.append(fold_accuracy)

        # Accuracy for each k parameter
        accuracies.append(accuracy_list)
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
