"""Script for running a K-nearest neighbors classifier."""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from final_project.modeling.run_missions import run_missions


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"


print()
# create K-nearest neighbors classifier:
model = KNeighborsClassifier()
# normalization type:
norm_type = "standard"
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 10

# hyperparameter values to search over:
k_values = np.arange(start=0, stop=51, step=2)
k_values[0] = 1
hyperparams = {"model__n_neighbors": k_values,
               "model__weights": ["uniform", "distance"]}
# scoring metric:
metric = "f1_macro"

# train and evaluate model:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, hyper_params=hyperparams, metric=metric,
             verbose=2)

