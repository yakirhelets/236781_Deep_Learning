import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        dist = torch.distributions.Normal(0, weight_std)
        self.weights = dist.sample((n_features, n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = x.mm(self.weights)
        y_pred = torch.as_tensor(class_scores.mode(dim=1)[-1])
        # ========================
        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        equal_elements = torch.sum(y == y_pred).type(torch.float32)
        acc = equal_elements / len(y)
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        train_loss = []
        train_accuracy = []

        valid_loss = []
        valid_accuracy = []

        print('Training', end='')
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======

            # dl_train_new = torch.utils.data.DataLoader(dataset=dl_train.dataset, batch_size=dl_train.batch_size,
            #                                            num_workers=dl_train.num_workers)
            # dl_valid_new = torch.utils.data.DataLoader(dataset=dl_valid.dataset, batch_size=dl_valid.batch_size,
            #                                            num_workers=dl_valid.num_workers)

            # Evaluation on the training set
            # acc_list = []
            # loss_list = []
            batch_num = 0
            for (x, y) in dl_train:
                y_pred, class_scores = self.predict(x)
                accuracy = LinearClassifier.evaluate_accuracy(y, y_pred)
                total_correct += accuracy
                # acc_list.append(accuracy)
                curr_loss = loss_fn.loss(x, y, class_scores, y_pred)
                average_loss += curr_loss
                # loss_list.append(curr_loss)

                # Computing the gradient
                grad = loss_fn.grad()
                # grad += (weight_decay * self.weights)
                # self.weights -= (learn_rate * grad)
                reg_term = self.weights * weight_decay
                self.weights = self.weights - (learn_rate * grad + reg_term)

                batch_num += 1

            train_accuracy += [total_correct / batch_num]
            train_loss += [average_loss / batch_num]
            # train_res[0].append(np.average(acc_list))
            # train_res[1].append(np.average(loss_list))

            # Evaluation on the validation set


            total_correct = 0
            average_loss = 0

            # acc_list = []
            # loss_list = []
            batch_num = 0
            for (x, y) in dl_valid:
                y_pred, class_scores = self.predict(x)
                accuracy = LinearClassifier.evaluate_accuracy(y, y_pred)
                total_correct += accuracy
                # acc_list.append(accuracy)
                curr_loss = loss_fn.loss(x, y, class_scores, y_pred)
                average_loss += curr_loss
                # loss_list.append(curr_loss)

                batch_num += 1

            valid_accuracy += [total_correct / batch_num]
            valid_loss += [average_loss / batch_num]
            # valid_res[0].append(np.average(acc_list))
            # valid_res[1].append(np.average(loss_list))

            # ========================
            print('.', end='')

        train_res = Result(accuracy=train_accuracy, loss=train_loss)
        valid_res = Result(accuracy=valid_accuracy, loss=valid_loss)

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        new_weights = None
        if has_bias:
            # remove first col
            new_weights = self.weights[1:,:]

        n_classes = new_weights.shape[1]
        C = img_shape[0]
        H = img_shape[1]
        W = img_shape[2]

        new_weights = new_weights.t()
        w_images = new_weights.view(n_classes, C, H, W)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0., learn_rate=0., weight_decay=0.)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.001
    hp['learn_rate'] = 0.01
    hp['weight_decay'] = 0.001
    # ========================

    return hp
