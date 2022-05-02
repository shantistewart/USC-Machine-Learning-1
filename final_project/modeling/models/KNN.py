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
# hyperparameter values to search over:
k_values = np.arange(start=0, stop=101, step=10)
k_values[0] = 1
hyperparams = {"model__n_neighbors": k_values}
# scoring metric:
metric = "f1_macro"

# train and evaluate model:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, hyper_params=hyperparams, scoring=metric,
             verbose=2)

