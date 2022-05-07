"""File containing class for trivial model."""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class TrivialClassifier(ClassifierMixin, BaseEstimator):
    """Class trivial model.

    Attributes:
        class_priors_: Class prior probabilities.
            dim: (n_classes, )
        Other attributes required to be a valid sklearn classifier.
    """

    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        """Fits trivial model by computing class prior probabilities.

        Args:
            X: Features.
                dim: (N, D)
            y: Labels.
                dim: (N, )

        Returns: self
        """

        # things required to be a valid sklearn classifier:
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True

        # compute class prior probabilities:
        class_vals, class_counts = np.unique(y, return_counts=True)
        self.class_priors_ = class_counts / np.sum(class_counts)

        return self

    def predict(self, X):
        """Predicts class labels of data.

        Args:
            X: Features.
                dim: (N, D)

        Returns:
            y_pred: Class label predictions.
                dim: (N, )
        """

        # things required to be a valid sklearn classifier:
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        check_array(X)

        # randomly make predictions based on class prior probabilities:
        y_pred = np.random.choice(self.classes_, p=self.class_priors_, size=X.shape[0])

        return y_pred

