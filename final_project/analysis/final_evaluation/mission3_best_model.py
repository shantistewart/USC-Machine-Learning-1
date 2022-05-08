"""Script for running the best-performing model for Mission 3 (support vector machine classifier)."""


from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
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
pca = True
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 5


# best hyperparameters of SVM:
# type of kernel to use for SVM:
kernel_type = ["rbf"]
# regularization parameter C in SVM (inversely proportional to regularization strength):
C = [6.22]
# coefficient for kernel:
gamma = [0.397]
# scoring function for SelectKBest feature selection:
kbest_score = [f_classif]
# number of features to keep for SelectKBest feature selection:
n_features_select = [1]
# fraction of variance required to be explained by result of PCA:
pca_var = [0.95]

hyperparams = {"model__kernel": kernel_type,
               "model__C": C,
               "model__gamma": gamma,
               "selector__score_func": kbest_score,
               "selector__k": n_features_select,
               "pca__svd_solver": ["full"],
               "pca__n_components": pca_var,
               "pca__whiten": [True]}

# run all missions:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, feature_select=feature_select, pca=pca,
             hyper_params=hyperparams, n_folds=n_folds, final_eval=True, missions=[3], verbose=2)

