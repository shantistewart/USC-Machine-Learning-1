"""Script for running a K-nearest neighbors classifier."""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from final_project.modeling.run_missions import run_missions


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"


print()
# create K-nearest neighbors classifier:
model = KNeighborsClassifier()
# normalization type:
norm_type = "standard"
# feature selection method:
feature_select = "KBest"
# whether to use PCA:
pca = False
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 10


# hyperparameter values to search over:
# number of nearest neighbors for KNN:
K_values = np.arange(start=0, stop=21, step=5)
K_values[0] = 1
# type of weight function for KNN:
weight_types = ["uniform", "distance"]
# scoring function for SelectKBest feature selection:
kbest_scores = [f_classif, mutual_info_classif]
# number of features to keep for SelectKBest feature selection:
n_features_select = np.arange(start=5, stop=41, step=5)

hyperparams = {"model__n_neighbors": K_values,
               "model__weights": weight_types,
               "selector__score_func": kbest_scores,
               "selector__k": n_features_select}
print("Hyperparameter search values:\n{}".format(hyperparams))
# scoring metric:
metric = "f1_macro"

# run all missions:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, feature_select=feature_select, pca=pca,
             hyper_params=hyperparams, metric=metric, verbose=2)

