"""File containing class for training and evaluation pipelines."""


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ModelPipeline:
    """Class for training and evaluation pipelines.

    Attributes:
        pipe: sklearn pipeline object.
    """

    def __init__(self):
        self.pipe = None

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
        self.pipe = Pipeline(steps=[
            ("normalizer", normalizer),
            ("model", model)
        ])

