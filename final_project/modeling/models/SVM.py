"""Script for running a support vector machine classifier."""


import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, mutual_info_classif
from final_project.modeling.run_missions import run_missions


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"


print()
# create support vector machine classifier:
model = SVC()
# normalization type:
norm_type = "standard"
# feature selection method:
feature_select = "KBest"
# whether to use PCA:
pca = False
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 5


# hyperparameter search method and scoring metric:
search_type = "random"
n_iters = 1000
metric = "f1_macro"
# hyperparameter values to search over:
# type of kernel to use for SVM:
kernel_types = ["rbf", "sigmoid"]
# regularization parameter C in SVM (inversely proportional to regularization strength):
C_values = np.logspace(-3, 3, num=100, base=2)
# coefficient for kernel:
gamma_values = np.logspace(-10, 3, num=100, base=2)
# scoring function for SelectKBest feature selection:
kbest_scores = [f_classif, mutual_info_classif]
# number of features to keep for SelectKBest feature selection:
n_features_select = np.arange(start=1, stop=45, step=1)

hyperparams = {"model__kernel": kernel_types,
               "model__C": C_values,
               "model__gamma": gamma_values,
               "selector__score_func": kbest_scores,
               "selector__k": n_features_select}

# run all missions:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, feature_select=feature_select, pca=pca,
             hyper_params=hyperparams, search_type=search_type, metric=metric, n_iters=n_iters, n_folds=n_folds,
             verbose=2)

