"""File containing class for training and evaluation pipelines."""


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from final_project.preprocessing.load_data_class import DataLoader
from final_project.preprocessing.preprocess_class import Preprocessor


class ModelPipeline:
    """Class for training and evaluation pipelines.

    Attributes:
        mission: (1, 2, or 3) Type of mission (task).
            Mission 1: predict G1 and remove G2 and G3 features.
            Mission 2: predict G3 and remove G1 and G2 features.
            Mission 3: predict G3, while keeping G1 and G2 features.
        pipe: sklearn pipeline object for model.
    """

    def __init__(self, mission):
        self.mission = mission
        self.model_pipe = None

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
            normalizer = StandardScaler

        # create pipeline:
        self.model_pipe = Pipeline(steps=[
            ("normalizer", normalizer),
            ("model", model)
        ])

    def train(self, train_data_file, model, norm_type="standard"):
        """Trains model.

        Args:
            train_data_file: Name of data file.
            model: sklearn model (estimator) object.
            norm_type: Type of normalization to use.
                allowed values: "standard"

        Returns: None
        """

        # load data:
        loader = DataLoader()
        X_orig, y_orig = loader.load_and_split_data(train_data_file, self.mission)
        # preprocess data:
        prep = Preprocessor()
        X_train, y_train = prep.preprocess_data(X_orig, y_orig, train=True)

        # make sklearn pipeline object:
        self.make_pipeline(model, norm_type=norm_type)
        # train model:
        self.model_pipe.fit(X_train, y_train)

