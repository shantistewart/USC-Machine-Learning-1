"""File containing class for training and evaluation pipelines."""


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from final_project.preprocessing.load_data_class import DataLoader
from final_project.preprocessing.preprocess_class import Preprocessor


class ModelPipeline:
    """Class for training and evaluation pipelines.

    Attributes:
        mission: (1, 2, or 3) Type of mission (task).
            Mission 1: predict G1 and remove G2 and G3 features.
            Mission 2: predict G3 and remove G1 and G2 features.
            Mission 3: predict G3, while keeping G1 and G2 features.
        preprocessor: Preprocessor object.
        model_pipe: sklearn Pipeline object for model.
    """

    def __init__(self, mission, model, norm_type="standard"):
        """Initializes ModelPipeline object.

        Args:
            mission: (1, 2, or 3) Type of mission (task).
                Mission 1: predict G1 and remove G2 and G3 features.
                Mission 2: predict G3 and remove G1 and G2 features.
                Mission 3: predict G3, while keeping G1 and G2 features.
            model: sklearn model (estimator) object.
            norm_type: Type of normalization to use.
                allowed values: "standard"

        Returns: None
        """

        self.mission = mission
        self.preprocessor = Preprocessor()
        # make sklearn pipeline object:
        self.model_pipe = None
        self.make_pipeline(model, norm_type=norm_type)

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

    def eval(self, data_file, verbose=1):
        """Evaluates model on data.

        Args:
            data_file: Name of data file.
            verbose: Nothing printed (0), accuracy and macro F1-score printed (1), all metrics printed (2)

        Returns:
            accuracy: Subset accuracy.
            macro_f1: Macro F1-score.
            conf_matrix: Confusion matrix.
        """

        # load data:
        loader = DataLoader()
        X_orig, y_orig = loader.load_and_split_data(data_file, self.mission)
        # preprocess data:
        X, y = self.preprocessor.preprocess_data(X_orig, y_orig, train=False)

        # predict on data:
        y_pred = self.model_pipe.predict(X)

        # compute metrics (subset accuracy, macro F1-score, confusion matrix):
        accuracy = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average="macro")
        conf_matrix = confusion_matrix(y, y_pred, normalize=None)
        # print metrics:
        if verbose == 1 or verbose == 2:
            print("accuracy = {} %".format(100*accuracy))
            print("macro F1-score = {}".format(macro_f1))
        if verbose == 2:
            print("confusion matrix = \n{}".format(conf_matrix))

        return accuracy, macro_f1, conf_matrix

    def make_pipeline(self, model, norm_type="standard"):
        """Creates a sklearn model pipeline object.

        Args:
            model: sklearn model (estimator) object.
            norm_type: Type of normalization to use.
                allowed values: "standard"

        Returns: None
        """

        # validate normalization type:
        if norm_type != "standard":
            raise Exception("Invalid normalization type.")

        normalizer = None
        # create sklearn normalization object:
        if norm_type == "standard":
            normalizer = StandardScaler()

        # create pipeline:
        self.model_pipe = Pipeline(steps=[
            ("normalizer", normalizer),
            ("model", model)
        ])

