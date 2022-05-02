"""Script for running a nearest means classifier (baseline model)."""


from sklearn.neighbors import NearestCentroid
from final_project.modeling.run_missions import run_missions


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"


print()
# create nearest means classifier (with Euclidean distance metric):
model = NearestCentroid(metric="euclidean")
# normalization type:
norm_type = "standard"
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 10

# train and evaluate model:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, tune_model=False, n_folds=n_folds,
             final_eval=True, verbose=2)

