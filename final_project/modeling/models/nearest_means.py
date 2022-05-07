"""Script for running a nearest-means classifier with feature engineering."""


import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import f_classif, mutual_info_classif
from final_project.modeling.run_missions import run_missions


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"
# number of features (minimum over all missions):
N_FEATURES = 43


print()
# create nearest means classifier (with Euclidean distance metric):
model = NearestCentroid(metric="euclidean")
# normalization type:
norm_type = "standard"
# feature selection method:
feature_select = "KBest"
# feature engineering type:
feature_eng = "poly"
# whether to use PCA:
pca = True
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 5


# hyperparameter search method and scoring metric:
search_type = "grid"
metric = "f1_macro"
# hyperparameter values to search over:
# scoring function for SelectKBest feature selection:
kbest_scores = [f_classif, mutual_info_classif]
# number of features to keep for SelectKBest feature selection:
n_features_select = np.arange(start=1, stop=int(np.round(0.5*N_FEATURES) + 1), step=1)
# maximum degree for polynomial feature engineering:
max_degrees = [2]
# fraction of variance required to be explained by result of PCA:
pca_var = np.arange(0.2, 1.05, step=0.1)
pca_var[len(pca_var)-1] = 0.99

if pca:
    hyperparams = {"selector__score_func": kbest_scores,
                   "selector__k": n_features_select,
                   "engineer__degree": max_degrees,
                   "engineer__include_bias": [False],
                   "pca__svd_solver": ["full"],
                   "pca__n_components": pca_var,
                   "pca__whiten": [True]}
else:
    hyperparams = {"selector__score_func": kbest_scores,
                   "selector__k": n_features_select,
                   "engineer__degree": max_degrees,
                   "engineer__include_bias": [False]}
print("Hyperparameter search values:\n{}".format(hyperparams))

# run all missions:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, feature_select=feature_select,
             feature_eng=feature_eng, pca=pca, hyper_params=hyperparams, search_type=search_type, metric=metric,
             n_folds=n_folds, verbose=2)

