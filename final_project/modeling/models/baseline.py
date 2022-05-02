"""Script for running a nearest means classifier (baseline system)."""


from sklearn.neighbors import NearestCentroid
from final_project.modeling.final_evaluation import final_eval


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"


print()
# create nearest means classifier (with Euclidean distance metric):
model = NearestCentroid(metric="euclidean")
# normalization type:
norm_type = "standard"

# train and evaluate model:
final_eval(model, norm_type, train_data_file, test_data_file)

