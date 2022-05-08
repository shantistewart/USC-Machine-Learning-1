"""Script for running the best-performing model for Mission 2 (nearest means classifier with feature engineering)."""


from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import f_classif
from final_project.modeling.run_missions import run_missions


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"


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


# best hyperparameters of nearest means classifier:
# scoring function for SelectKBest feature selection:
kbest_score = [f_classif]
# number of features to keep for SelectKBest feature selection:
n_features_select = [18]
# maximum degree for polynomial feature engineering:
max_degree = [2]
# fraction of variance required to be explained by result of PCA:
pca_var = [0.7]

hyperparams = {"selector__score_func": kbest_score,
               "selector__k": n_features_select,
               "engineer__degree": max_degree,
               "engineer__include_bias": [False],
               "pca__svd_solver": ["full"],
               "pca__n_components": pca_var,
               "pca__whiten": [True]}

# run all missions:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, feature_select=feature_select,
             feature_eng=feature_eng, pca=pca, hyper_params=hyperparams, n_folds=n_folds, final_eval=True,
             missions=[2], verbose=2)

