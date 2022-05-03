"""Script for running a K-nearest neighbors classifier."""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from final_project.modeling.run_missions import run_missions


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"
# number of features (minimum over all missions):
N_FEATURES = 43


print()
# create K-nearest neighbors classifier:
model = KNeighborsClassifier()
# normalization type:
norm_type = "standard"
# feature selection method:
feature_select = "KBest"
# whether to use PCA:
pca = True
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 5


# hyperparameter search method and scoring metric:
search_type = "random"
n_iters = 1000
metric = "f1_macro"
# hyperparameter values to search over:
# type of weight function for KNN:
weight_types = ["uniform", "distance"]
# number of nearest neighbors for KNN:
K_values = np.arange(start=1, stop=31, step=1)
# scoring function for SelectKBest feature selection:
kbest_scores = [f_classif, mutual_info_classif]
# number of features to keep for SelectKBest feature selection:
n_features_select = np.arange(start=1, stop=N_FEATURES+1, step=1)
# fraction of variance required to be explained by result of PCA:
pca_var = np.linspace(0.25, 1.0, num=16)
pca_var[len(pca_var)-1] = 0.99

if pca:
    hyperparams = {"model__weights": weight_types,
                   "model__n_neighbors": K_values,
                   "selector__score_func": kbest_scores,
                   "selector__k": n_features_select,
                   "pca__svd_solver": ["full"],
                   "pca__n_components": pca_var,
                   "pca__whiten": [True]}
else:
    hyperparams = {"model__weights": weight_types,
                   "model__n_neighbors": K_values,
                   "selector__score_func": kbest_scores,
                   "selector__k": n_features_select}
print("Hyperparameter search values:\n{}".format(hyperparams))

# run all missions:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, feature_select=feature_select, pca=pca,
             hyper_params=hyperparams, search_type=search_type, metric=metric, n_iters=n_iters, n_folds=n_folds,
             verbose=2)

