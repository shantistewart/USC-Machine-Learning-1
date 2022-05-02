"""File containing function for running final model evaluation."""


from final_project.modeling.model_pipeline_class import ModelPipeline


# default mission values:
MISSIONS = [1, 2, 3]


def final_eval(model, norm_type, train_data_file, test_data_file, missions=None, verbose=2):
    """Trains final model on full training set and evaluates on test set, for each mission.

    Args:
        model: sklearn model (estimator) object.
        norm_type: Type of normalization to use.
            allowed values: "standard"
        train_data_file: Name of training data file.
        test_data_file: Name of test data file.
        missions: List of missions to perform.
        verbose: Nothing printed (0), accuracy and macro F1-score printed (1), all metrics printed (2).

    Returns:
        metrics: Nested dictionaries of training/test set metrics.
            metrics["mission_x"]["train"]["metric_name"] = mission x's training set metric_name metric
            metrics["mission_x"]["test"]["metric_name"] = mission x's test set metric_name metric
    """

    # set default missions:
    if missions is None:
        missions = MISSIONS

    metrics = {}
    # run final model evaluation for all missions:
    for mission in missions:
        if verbose != 0:
            print("\nMISSION {}:\n".format(mission))

        # train model:
        model_pipe = ModelPipeline(mission, model, norm_type=norm_type)
        if verbose != 0:
            print("Training model...")
        model_pipe.train(train_data_file)

        # evaluate model on training set:
        if verbose != 0:
            print("\nEvaluating on training set...")
        train_metrics = model_pipe.eval(train_data_file, verbose=verbose)
        # evaluate model on test set:
        if verbose != 0:
            print("\nEvaluating on test set...")
        test_metrics = model_pipe.eval(test_data_file, verbose=verbose)
        if verbose != 0:
            print("")

        # save training/test set metrics to nested dictionary:
        metrics["mission_" + str(mission)] = {}
        metrics["mission_" + str(mission)]["train"] = train_metrics
        metrics["mission_" + str(mission)]["test"] = test_metrics

    return metrics

