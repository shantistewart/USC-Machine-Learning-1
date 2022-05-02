"""File containing function for running final model evaluation."""


from final_project.modeling.model_pipeline_class import ModelPipeline


# default mission values:
MISSIONS = [1, 2, 3]


def final_eval(model, norm_type, train_data_file, test_data_file, missions=None):
    """Trains final model on full training set and evaluates on test set, for each mission.

    Args:
        model: sklearn model (estimator) object.
        norm_type: Type of normalization to use.
            allowed values: "standard"
        train_data_file: Name of training data file.
        test_data_file: Name of test data file.
        missions: List of missions to perform.

    Returns:
        train_metrics: Dictionary of training set metrics.
        test_metrics: Dictionary of test set metrics.
    """

    # set default missions:
    if missions is None:
        missions = MISSIONS

    # run final model evaluation for all missions:
    for mission in missions:
        print("\nMISSION {}:".format(mission))

        # train model:
        model_pipe = ModelPipeline(mission, model, norm_type=norm_type)
        print("Training model...")
        model_pipe.train(train_data_file)

        # evaluate model on training set:
        print("Evaluating on training set...")
        model_pipe.eval(train_data_file, verbose=2)

        # evaluate model on test set:
        print("Evaluating on test set...")
        model_pipe.eval(test_data_file, verbose=2)

