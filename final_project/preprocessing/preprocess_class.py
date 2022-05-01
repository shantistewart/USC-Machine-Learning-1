"""File containing class for preprocessing data."""


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class Preprocessor:
    """Class for preprocessing data.

    Attributes:
        ordinal_encoder: sklearn OrdinalEncoder object.
        one_hot_encoder: sklearn OneHotEncoder object.
    """

    def __init__(self):
        self.ordinal_encoder = None
        self.one_hot_encoder = None

    def preprocess_data(self, X_orig, y_orig, bin_feature_names, nom_feature_names, bins, train):
        """Encodes categorical features and quantizes labels into bins (for classification).

        Args:
            X_orig: Original features (dataframe).
                dim: (N_orig, D)
            y_orig: Original labels (series).
                dim: (N, )
            bin_feature_names: List of names of binary (categorical with two possible values) features.
            nom_feature_names: List of names of nominal (multivalued categorical with no logical ordering) features.
            bins: 1D array of bin edges (decreasing order), where bin i = [bins[i], bin[i-1]).
                dim: (n_classes+1, )
            train: Whether in training mode or not.

        Returns:
            X: Encoded features.
                dim: (N, D)
            y: Quantized labels, with integer encoding (values in [0, n_classes-1])
                dim: (N, )
        """

        # encode features and convert to numpy array:
        X = self.encode_features(X_orig, bin_feature_names, nom_feature_names, train)
        X = X.to_numpy(dtype=float, copy=True)

        # quantize labels:
        y = self.quantize_labels(y_orig, bins)

        return X, y

    def encode_features(self, X_orig, bin_feature_names, nom_feature_names, train):
        """Encodes categorical features.

        Args:
            X_orig: Original features (dataframe).
                dim: (N_orig, D)
            bin_feature_names: List of names of binary (categorical with two possible values) features.
            nom_feature_names: List of names of nominal (multivalued categorical with no logical ordering) features.
            train: Whether in training mode or not.

        Returns:
            X: Encoded features (dataframe).
                dim: (N, D)
        """

        # encode binary features using ordinal encoding:
        X_bin_orig = X_orig[bin_feature_names]
        self.ordinal_encoder = OrdinalEncoder(dtype=float)
        # fit encoder to data if in training mode:
        if train:
            self.ordinal_encoder.fit(X_bin_orig)
        # encode features and convert to dataframe:
        X_bin = self.ordinal_encoder.transform(X_bin_orig)
        X_bin = pd.DataFrame(data=X_bin, columns=X_bin_orig.columns)

        # encode nominal features using one-hot encoding:
        X_nom_orig = X_orig[nom_feature_names]
        self.one_hot_encoder = OneHotEncoder(sparse=False, dtype=float)
        # fit encoder to data if in training mode:
        if train:
            self.one_hot_encoder.fit(X_nom_orig)
        # encode features:
        X_nom = self.one_hot_encoder.transform(X_nom_orig)
        # generate column labels for encoded features by looping through all nominal features and their categories:
        encode_col_names = []
        for j in range(len(nom_feature_names)):
            for category in self.one_hot_encoder.categories_[j]:
                encode_col_names.append(nom_feature_names[j] + "--" + category)
        X_nom = pd.DataFrame(data=X_nom, columns=encode_col_names)

        # replace original binary features with encoded versions:
        X_orig[bin_feature_names] = X_bin

        # replace original nominal features with encoded versions:
        X = pd.concat([X_orig.drop(columns=nom_feature_names), X_nom], axis="columns")

        # convert entire dataframe to floats:
        X = X.astype(dtype=float)

        return X

    def quantize_labels(self, y_orig, bins):
        """Quantizes labels into bins, for classification.

        Args:
            y_orig: Original labels (series).
                dim: (N, )
            bins: 1D array of bin edges (decreasing order), where bin i = [bins[i], bin[i-1]).
                dim: (n_classes+1, )

        Returns:
            y: Quantized labels, with integer encoding (values in [0, n_classes-1])
                dim: (N, )
        """

        # convert series to numpy array:
        y_orig = y_orig.to_numpy(dtype=int, copy=True)
        # quantize original labels by binning them:
        y = np.digitize(y_orig, bins, right=False) - 1

        return y


"""
# TESTING:
from load_data_class import DataLoader

# data file name:
data_file = "../data/student_performance_train.csv"
# names of binary/nominal features:
bin_feature_names = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities",
                     "nursery", "higher", "internet", "romantic"]
nom_feature_names = ["Mjob", "Fjob", "reason", "guardian"]

print()
# get data:
get_data = DataLoader()
X_orig, y_orig = get_data.load_and_split_data(data_file, 3)
# print("Dim of X_orig: {}".format(X_orig.shape))
# print(X_orig.head())

# encode features:
prep = Preprocessor()
X = prep.encode_features(X_orig, bin_feature_names, nom_feature_names, train=True)
print()
print("Dim of X: {}".format(X.shape))
print(X.head())
print("\n")
print(X.info())
print()
print(X.describe())

# test quantize_labels() method:
bins = np.array([20.5, 15.5, 13.5, 11.5, 9.5, -0.5])
y = prep.quantize_labels(y_orig, bins)
print()
print(y_orig)
print()
print(y)

"""

