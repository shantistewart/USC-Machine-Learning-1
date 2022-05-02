"""File containing function for running missions."""


from final_project.modeling.model_pipeline_class import ModelPipeline


# default mission values:
MISSIONS = [1, 2, 3]


def run_missions(train_data_file, test_data_file, model, norm_type="standard", tune_model=True, hyper_params=None,
                 search_type="grid", n_folds=10, metric="accuracy", final_eval=False, missions=None, verbose=2):
    """Trains, tunes, and evaluates model, for each mission.

    Args:
        train_data_file: Name of training data file.
        test_data_file: Name of test data file.
        model: sklearn model (estimator) object, with some initial hyperparameters.
        norm_type: Type of normalization to use.
            allowed values: "standard"
        tune_model: Selects whether to tune hyperparameters of model.
        hyper_params: Dictionary of hyperparameter values to search over (ignored if tune_model = False).
        search_type: Hyperparameter search type (ignored if tune_model = False).
            allowed values: "grid"
        n_folds: Number of folds (K) to use in stratified K-fold cross validation (ignored if tune_model = False).
        metric: Type of metric to use for model evaluation (ignored if tune_model = False).
        final_eval: Selects whether to evaluate final model on test set.
        missions: List of missions to perform.
        verbose: Nothing printed (0), some things printed (1), everything printed (2).

    Returns:
        metrics: Nested dictionary of training/test set metrics.
            metrics["mission_x"]["train"]["metric_name"] = mission x's training set metric_name metric
            metrics["mission_x"]["test"]["metric_name"] = mission x's test set metric_name metric
        best_models: Nested dictionary of best model information after hyperparameter tuning.
            best_models["mission_x"]["hyperparams"] = mission x's best model hyperparameters
            best_models["mission_x"]["cv_score"] = mission x's best model cross-validation score.
    """

    # set default missions:
    if missions is None:
        missions = MISSIONS

    metrics = {}
    best_models = {}
    # run missions:
    for mission in missions:
        if verbose != 0:
            print("\nMISSION {}:\n".format(mission))

        # create initial ModelPipeline object:
        model_pipe = ModelPipeline(mission, model, norm_type=norm_type)

        # tune hyperparameters if selected:
        if tune_model:
            if verbose != 0:
                print("Tuning model hyperparameters...")
            if hyper_params is None:
                raise Exception("Hyperparameter search values argument is none.")
            best_model, best_hyperparams, best_cv_score = model_pipe.tune_hyperparams(train_data_file, hyper_params,
                                                                                      search_type=search_type,
                                                                                      n_folds=n_folds, metric=metric,
                                                                                      verbose=verbose)
            # save best model information to nested dictionary:
            best_models["mission_" + str(mission)] = {}
            best_models["mission_" + str(mission)]["hyperparams"] = best_hyperparams
            best_models["mission_" + str(mission)]["cv_score"] = best_cv_score

        # (re)train model on full training set:
        if verbose != 0:
            print("Training model on full training set...")
        model_pipe.train(train_data_file)

        # evaluate model on training set:
        if verbose != 0:
            print("\nEvaluating model on training set...")
        train_metrics = model_pipe.eval(train_data_file, verbose=verbose)
        # evaluate model on test set, if selected:
        if final_eval:
            if verbose != 0:
                print("\nEvaluating model on test set...")
            test_metrics = model_pipe.eval(test_data_file, verbose=verbose)
            if verbose != 0:
                print("")

        # save training/test set metrics to nested dictionary:
        metrics["mission_" + str(mission)] = {}
        metrics["mission_" + str(mission)]["train"] = train_metrics
        if final_eval:
            metrics["mission_" + str(mission)]["test"] = test_metrics

    return metrics, best_models

