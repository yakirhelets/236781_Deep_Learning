import sklearn

import numpy as np
import scipy # TODO maybe remove later
from pandas import DataFrame
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        # Bias trick already operated
        W = self.weights_.reshape((-1, 1)).transpose()
        x_t = X.transpose()
        y_pred_tmp = np.matmul(W, x_t)
        y_pred = y_pred_tmp[0]

        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution
        #  Use only numpy functions. Don't forget regularization.

        w_opt = None
        # ====== YOUR CODE: ======
        N = X.shape[0]
        first_term = np.matmul(X.transpose(), X) # X_t * X
        reg_matrix = self.reg_lambda * N * np.identity(X.shape[1]) # Regularization term
        reg_matrix[0][0] = 0
        mid_term = first_term + reg_matrix # Add regularization term
        second_term = np.linalg.inv(mid_term) # (X_t * X)^-1
        third_term = np.matmul(second_term, X.transpose()) # (X_t * X)^-1 * X_t
        w_opt = np.matmul(third_term, y) # (X_t * X)^-1 * X_t * y

        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        N = X.shape[0]
        ones = np.ones((N,1), dtype=X.dtype)
        xb = np.hstack((ones, X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        self.poly = PolynomialFeatures(degree)
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        X_all_polys = self.poly.fit_transform(X)
        X_transformed = X_all_polys
        # X_transformed = X_all_polys[:, [0,1,3,4,5,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,54,55,56,57,58,59,60,61,62,63,64,65,77,82,84,92,117,119]]
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    top_n_features = []
    top_n_corr = []
    features_corr_dict = {}
    # for each feature that is not the target feature, calc the p_x,y and add to a dict
    # with names as keys and corrs as values
    for col in df.columns:
        if col != target_feature:
            p = scipy.stats.pearsonr(df[target_feature], df[col])
            features_corr_dict[col] = p[0]

    for i in range(n):
        next_top = max(features_corr_dict, key=lambda y: np.abs(features_corr_dict[y])) # get top
        top_n_features.append(next_top) # insert name
        top_n_corr.append(features_corr_dict[next_top]) # insert value
        features_corr_dict.pop(next_top) # remove top from dict
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    mse = ((y_pred - y)**2).mean()
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    nominator = 0
    denominator = 0
    y_mean = np.mean(y)
    for i in range(len(y)):
        nominator += (y[i]-y_pred[i])**2
        denominator += (y[i]-y_mean)**2
    r2 = 1 - (nominator/denominator)
    # ========================
    return r2


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    params_to_try_dict = {'bostonfeaturestransformer__degree': degree_range, 'linearregressor__reg_lambda': lambda_range}
    grid_search_cv = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=params_to_try_dict,
                                                                iid=True, cv=k_folds)
    grid_search_cv.fit(X, y)
    best_params = grid_search_cv.best_params_
    # ========================

    return best_params
