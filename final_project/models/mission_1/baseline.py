"""Script for running a nearest means classifier (baseline system) for mission 1."""


import numpy as np
import matplotlib.pyplot as plt
from final_project.preprocessing.load_data_class import DataLoader
from final_project.preprocessing.preprocess_class import Preprocessor
from final_project.models.make_pipeline import make_pipeline


# training data file name:
train_data_file = "../../data/student_performance_train.csv"
# mission type:
mission = 1
# names of binary/nominal features:
bin_feature_names = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities",
                     "nursery", "higher", "internet", "romantic"]
nom_feature_names = ["Mjob", "Fjob", "reason", "guardian"]
# bins for quantizing labels:
bins = np.array([20.5, 16, 14, 12, 10, 0])


# TRAINING:

# load data:
loader = DataLoader()
X_orig, y_orig = loader.load_and_split_data(train_data_file, mission)
# preprocess data:
prep = Preprocessor()
X, y = prep.preprocess_data(X_orig, y_orig, bin_feature_names, nom_feature_names, bins, train=True)

# make model pipeline:

