"""File containing class for loading and splitting data into features and labels."""


import pandas as pd


class DataLoader:
    """Class for loading and splitting data into features and labels.

    Attributes:
        N: Total number of data points.
        D: Total number of features.
    """

    def __init__(self):
        self.N = None
        self.D = None

    def load_and_split_data(self, data_file, mission):
        """Loads and splits data into features and labels.

        Args:
            data_file: Name of data file.
            mission: (1, 2, or 3) Type of mission (task).
                Mission 1: predict G1 and remove G2 and G3 features.
                Mission 2: predict G3 and remove G1 and G2 features.
                Mission 3: predict G3, while keeping G1 and G2 features.

        Returns:
            X: Features (dataframe).
                dim: (N, D)
            y: Labels (series).
                dim: (N, )
        """

        # load data:
        data = self.load_data(data_file)
        # split data into features and labels according to mission type:
        X, y = self.split_data(data, mission)

        return X, y

    def load_data(self, data_file):
        """Loads data into a pandas dataframe.

        Args:
            data_file: Name of data file.

        Returns:
            data: Pandas dataframe containing data.
        """

        data = pd.read_csv(data_file)
        self.N = data.shape[0]

        return data

    def split_data(self, data, mission):
        """Splits data into features and labels according to mission type.

        Args:
            data: Pandas dataframe containing data.
            mission: (1, 2, or 3) Type of mission (task).
                Mission 1: predict G1 and remove G2 and G3 features.
                Mission 2: predict G3 and remove G1 and G2 features.
                Mission 3: predict G3, while keeping G1 and G2 features.

        Returns:
            X: Features (dataframe).
                dim: (N, D)
            y: Labels (series).
                dim: (N, )
        """

        # validate mission type:
        if mission != 1 and mission != 2 and mission != 3:
            raise Exception("Invalid mission type.")

        # split data into features and labels according to mission type:
        if mission == 1:
            # extract labels:
            y = data["G1"]
            # extract features:
            X = data.drop(columns=["G1", "G2", "G3"])
        elif mission == 2:
            # extract labels:
            y = data["G3"]
            # extract features:
            X = data.drop(columns=["G3", "G1", "G2"])
        elif mission == 3:
            # extract labels:
            y = data["G3"]
            # extract features:
            X = data.drop(columns=["G3"])
        # save number of features:
        self.D = y.shape[0]

        return X, y

