"""Script for running a nearest means classifier (baseline system) for mission 1."""


from sklearn.neighbors import NearestCentroid
from final_project.models.model_pipeline_class import ModelPipeline


# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"
# mission type:
mission = 1


print()
# create nearest means classifier (with Euclidean distance metric):
model = NearestCentroid(metric="euclidean")

# train model:
model_pipe = ModelPipeline(mission, model, norm_type="standard")
print("Training model...")
model_pipe.train(train_data_file)

# evaluate model on training set:
print("\nEvaluating on training set...")
model_pipe.eval(train_data_file, verbose=2)

# evaluate model on test set:
print("\nEvaluating on test set...")
model_pipe.eval(test_data_file, verbose=2)

