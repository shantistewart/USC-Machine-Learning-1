"""File containing function for running missions."""


from final_project.modeling.model_pipeline_class import ModelPipeline


# default mission values:
MISSIONS = [1, 2, 3]


def run_missions(train_data_file, test_data_file, model, norm_type="standard", feature_select=None, feature_eng=None,
                 pca=False, tune_model=True, hyper_params=None, search_type="grid", metric="accuracy", n_iters=None,
                 n_folds=10, final_eval=False, missions=None, verbose=2):
    """Trains, tunes, and evaluates model, for each mission.

    Args:
        train_data_file: Name of training data file.
        test_data_file: Name of test data file.
        model: sklearn model (estimator) object, with some initial hyperparameters.
        norm_type: Type of normalization to use.
            allowed values: "standard", None
        feature_select: Method of feature selection.
            allowed values: "KBest", "SFS", None
        feature_eng: Method of feature engineering.
            allowed values: "poly", None
        pca: Selects whether to use PCA.
        tune_model: Selects whether to tune hyperparameters of model.
        hyper_params: Dictionary of hyperparameter values to search over (ignored if tune_model = False).
        search_type: Hyperparameter search type (ignored if tune_model = False).
            allowed values: "grid", "random"
        metric: Type of metric to use for model evaluation (ignored if tune_model = False).
        n_iters: Number of hyperparameter combinations that are tried in random search (ignored if tune_model = False
            or if search_type != "random")
        n_folds: Number of folds (K) to use in stratified K-fold cross validation.
        final_eval: Selects whether to evaluate final model on test set.
        missions: List of missions to perform.
        verbose: Nothing printed (0), some things printed (1), everything printed (2).

    Returns:
        metrics: Nested dictionary of training/test set metrics.
            metrics["mission_x"]["train"]["metric_name"] = mission x's training set metric_name metric
            metrics["mission_x"]["val"]["metric_name"] = mission x's cross-validation metric_name metric
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
        model_pipe = ModelPipeline(mission, model, norm_type=norm_type, feature_select=feature_select,
                                   feature_eng=feature_eng, pca=pca)

        # if tuning not desired, train model using given hyperparameters
        #   (note: feature selection and PCA transformer are created with default parameters):
        if not tune_model:
            if verbose != 0:
                print("Training model...")
            model_pipe.train(train_data_file)
        # otherwise, tune hyperparameters:
        else:
            if verbose != 0:
                print("Tuning model hyperparameters...")
            if hyper_params is None:
                raise Exception("Hyperparameter search values argument is none.")
            best_model, best_hyperparams, best_cv_score = model_pipe.tune_hyperparams(train_data_file, hyper_params,
                                                                                      search_type=search_type,
                                                                                      metric=metric, n_iters=n_iters,
                                                                                      n_folds=n_folds, verbose=verbose)
            # save best model information to nested dictionary:
            best_models["mission_" + str(mission)] = {}
            best_models["mission_" + str(mission)]["hyperparams"] = best_hyperparams
            best_models["mission_" + str(mission)]["cv_score"] = best_cv_score

        # evaluate model on training set:
        if verbose != 0:
            print("\nEvaluating model on training set...")
        train_metrics = model_pipe.eval(train_data_file, "train", verbose=verbose)

        # evaluate model using cross validation:
        if verbose != 0:
            print("\nEvaluating model using cross validation...")
        val_metrics = model_pipe.eval(train_data_file, "cross_val", n_folds=n_folds, verbose=verbose)
        # (re)train model on full training set (since cross validation alters the sklearn Pipeline object):
        model_pipe.train(train_data_file)

        # evaluate model on test set, if selected:
        if final_eval:
            if verbose != 0:
                print("\nEvaluating model on test set...")
            test_metrics = model_pipe.eval(test_data_file, "test", verbose=verbose)
        if verbose != 0:
            print("")

        # save training/cross-validation/test set metrics to nested dictionary:
        metrics["mission_" + str(mission)] = {}
        metrics["mission_" + str(mission)]["train"] = train_metrics
        metrics["mission_" + str(mission)]["val"] = val_metrics
        if final_eval:
            metrics["mission_" + str(mission)]["test"] = test_metrics

    return metrics, best_models

