"""Script for running trivial model."""


from final_project.modeling.models.trivial_model_class import TrivialClassifier
from final_project.modeling.model_pipeline_class import ModelPipeline


# default mission values:
MISSIONS = [1, 2, 3]
# data file names:
train_data_file = "../../data/student_performance_train.csv"
test_data_file = "../../data/student_performance_test.csv"


print()
# create trivial model:
model = TrivialClassifier()
# normalization type:
norm_type = "standard"
# number of folds (K) to use in stratified K-fold cross validation:
n_folds = 10
# number of trials for averaging performance:
n_trials = 10

# run all missions:
verbose = 1
for mission in MISSIONS:
    if verbose != 0:
        print("\nMISSION {}:\n".format(mission))

    # create initial ModelPipeline object:
    model_pipe = ModelPipeline(mission, model, norm_type=norm_type)

    # train model:
    if verbose != 0:
        print("Training model...")
    model_pipe.train(train_data_file)

    accuracy_sum_train = 0.0
    macro_f1_sum_train = 0.0
    accuracy_sum_val = 0.0
    macro_f1_sum_val = 0.0
    for i in range(n_trials):
        # evaluate model on training set:
        train_metrics = model_pipe.eval(train_data_file, "train", verbose=0)
        accuracy_sum_train += train_metrics["accuracy"]
        macro_f1_sum_train += train_metrics["macro_f1"]

        # evaluate model using cross validation:
        val_metrics = model_pipe.eval(train_data_file, "cross_val", n_folds=n_folds, verbose=0)
        accuracy_sum_val += val_metrics["accuracy"]
        macro_f1_sum_val += val_metrics["macro_f1"]

    # compute averages:
    accuracy_train = accuracy_sum_train / n_trials
    macro_f1_train = macro_f1_sum_train / n_trials
    accuracy_val = accuracy_sum_val / n_trials
    macro_f1_val = macro_f1_sum_val / n_trials

    print()
    print("Average training accuracy = {}".format(100 * accuracy_train))
    print("Average training macro F1-score = {}".format(macro_f1_train))
    print("Average cross-validation accuracy = {}".format(100 * accuracy_val))
    print("Average cross-validation macro F1-score = {}".format(macro_f1_val))

