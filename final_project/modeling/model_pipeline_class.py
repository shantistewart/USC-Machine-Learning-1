"""File containing class for training, tuning, and evaluating a model."""


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from final_project.preprocessing.load_data_class import DataLoader
from final_project.preprocessing.preprocess_class import Preprocessor


class ModelPipeline:
    """Class for training, tuning, and evaluating a model.

    Attributes:
        mission: (1, 2, or 3) Type of mission (task).
            Mission 1: predict G1 and remove G2 and G3 features.
            Mission 2: predict G3 and remove G1 and G2 features.
            Mission 3: predict G3, while keeping G1 and G2 features.
        preprocessor: Preprocessor object.
        norm_type: Type of normalization to use.
            allowed values: "standard"
        feature_select: Method of feature selection.
            allowed values: "KBest", "SFS", or None
        pca: Selects whether to use PCA.
        model_pipe: sklearn Pipeline object for model.
            Updated to best model (i.e., model with best hyperparameters) when tune_hyperparams() method is called.
        best_hyperparams: Best hyperparameters (dictionary).
    """

    def __init__(self, mission, model, norm_type="standard", feature_select=None, pca=False):
        """Initializes ModelPipeline object.

        Args:
            mission: (1, 2, or 3) Type of mission (task).
                Mission 1: predict G1 and remove G2 and G3 features.
                Mission 2: predict G3 and remove G1 and G2 features.
                Mission 3: predict G3, while keeping G1 and G2 features.
            model: sklearn model (estimator) object.
            norm_type: Type of normalization to use.
                allowed values: "standard"
            feature_select: Method of feature selection.
                allowed values: "KBest", "SFS", or None
            pca: Selects whether to use PCA.

        Returns: None
        """

        # validate normalization type and feature selection method:
        if norm_type != "standard":
            raise Exception("Invalid normalization type.")
        if feature_select != "KBest" and feature_select != "SFS":
            raise Exception("Invalid feature selection method.")

        self.mission = mission
        self.preprocessor = Preprocessor()
        self.norm_type = norm_type
        self.feature_select = feature_select
        self.pca = pca
        self.best_hyperparams = None

        # initialize sklearn pipeline object:
        self.model_pipe = None
        self.make_pipeline(model)

    def train(self, train_data_file):
        """Trains model.

        Args:
            train_data_file: Name of training data file.

        Returns: None
        """

        # load data:
        loader = DataLoader()
        X_orig, y_orig = loader.load_and_split_data(train_data_file, self.mission)
        # preprocess data:
        X_train, y_train = self.preprocessor.preprocess_data(X_orig, y_orig, train=True)

        # train model:
        self.model_pipe.fit(X_train, y_train)

    def eval(self, data_file, eval_type, n_folds=10, verbose=1):
        """Evaluates model on data.

        Args:
            data_file: Name of data file.
            eval_type: Type of evaluation to run.
                allowed values: "train", "cross_val", "test"
            n_folds: Number of folds (K) to use in stratified K-fold cross validation (ignored if eval_type !=
                "cross_val").
            verbose: Nothing printed (0), accuracy and macro F1-score printed (1), all metrics printed (2).

        Returns:
            metrics: Dictionary of performance metrics.
                metrics["accuracy"] = subset accuracy
                metrics["macro_f1"] = macro F1-score
                metrics["conf_matrix"] = confusion matrix
        """

        # validate evaluation type:
        if eval_type != "train" and eval_type != "cross_val" and eval_type != "test":
            raise Exception("Invalid evaluation type.")

        # load data:
        loader = DataLoader()
        X_orig, y_orig = loader.load_and_split_data(data_file, self.mission)
        # preprocess data:
        if eval_type == "test":
            train = False
        else:
            train = True
        X, y = self.preprocessor.preprocess_data(X_orig, y_orig, train=train)

        # perform cross validation if selected:
        if eval_type == "cross_val":
            scores = cross_validate(self.model_pipe, X, y, cv=n_folds, scoring=["accuracy", "f1_macro"])
            accuracy = np.mean(scores["test_accuracy"])
            macro_f1 = np.mean(scores["test_f1_macro"])
        # otherwise, evaluate directly on provided dataset:
        else:
            # predict on data:
            y_pred = self.model_pipe.predict(X)
            # compute metrics (subset accuracy, macro F1-score, confusion matrix):
            accuracy = accuracy_score(y, y_pred)
            macro_f1 = f1_score(y, y_pred, average="macro")
            if eval_type == "test":
                conf_matrix = confusion_matrix(y, y_pred, normalize=None)
        # print metrics:
        if verbose != 0:
            print("accuracy = {} %".format(100*accuracy))
            print("macro F1-score = {}".format(macro_f1))
        if eval_type == "test" and verbose == 2:
            print("confusion matrix = \n{}".format(conf_matrix))

        # save metrics to dictionary:
        metrics = {"accuracy": accuracy,
                   "macro_f1": macro_f1}
        if eval_type == "test":
            metrics["conf_matrix"] = conf_matrix

        return metrics

    def tune_hyperparams(self, data_file, hyper_params, search_type="grid", n_folds=10, metric="accuracy", verbose=1):
        """Tunes hyperparameters (i.e., model selection).

        Args:
            data_file: Name of data file (used for training/validation).
            hyper_params: Dictionary of hyperparameter values to search over.
            search_type: Hyperparameter search type.
                allowed values: "grid"
            n_folds: Number of folds (K) to use in stratified K-fold cross validation.
            metric: Type of metric to use for model evaluation.
            verbose: Nothing printed (not 2), best hyperparameters printed (2).

        Returns:
            model_pipe: sklearn Pipeline object for best model (i.e., model with best hyperparameters).
            best_hyperparams: Best hyperparameters (dictionary).
            best_cv_score: Cross-validation score of best model.
        """

        # validate hyperparameter search type:
        if search_type != "grid":
            raise Exception("Invalid hyperparameter search type.")

        # load data:
        loader = DataLoader()
        X_orig, y_orig = loader.load_and_split_data(data_file, self.mission)
        # preprocess data:
        X, y = self.preprocessor.preprocess_data(X_orig, y_orig, train=True)

        # tune hyperparameters:
        search = None
        if search_type == "grid":
            search = GridSearchCV(self.model_pipe, hyper_params, cv=n_folds, scoring=metric)
        search.fit(X, y)

        # save best model, best hyperparameters, and best cross-validation score:
        self.model_pipe = search.best_estimator_
        self.best_hyperparams = search.best_params_
        best_cv_score = search.best_score_
        # print hyperparameter tuning results:
        if verbose == 2:
            print("Best hyperparameters: {}".format(self.best_hyperparams))

        return self.model_pipe, self.best_hyperparams, best_cv_score

    def make_pipeline(self, model):
        """Creates a sklearn Pipeline object for model.

        Args:
            model: sklearn model (estimator) object.

        Returns: None
        """

        # create sklearn normalization transformer:
        normalizer = None
        if self.norm_type == "standard":
            normalizer = StandardScaler()
        # create sklearn feature selection transformer:
        feature_selector = None
        if self.feature_select == "KBest":
            feature_selector = SelectKBest()
        elif self.feature_select == "SFS":
            feature_selector = SequentialFeatureSelector()
        # create sklearn PCA transformer:
        pca = None
        if pca:
            pca = PCA()

        # create pipeline:
        if pca:
            self.model_pipe = Pipeline(steps=[
                ("normalizer", normalizer),
                ("model", model),
                ("feature_selector", feature_selector),
                ("pca", pca)
            ])
        else:
            self.model_pipe = Pipeline(steps=[
                ("normalizer", normalizer),
                ("model", model),
                ("feature_selector", feature_selector)
            ])

