"""Script for running a nearest means classifier (baseline system)."""


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

# train and evaluate model:
run_missions(train_data_file, test_data_file, model, norm_type=norm_type, tune_model=False, verbose=2)

