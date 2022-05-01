"""File containing function for making a sklearn model pipeline."""


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(model, norm_type="standard"):
    """Creates a sklearn model pipeline object.

    Args:
        model: sklearn model (estimator) object.
        norm_type: Type of normalization to use.
            allowed values: "standard"

    Returns:
        pipe: sklearn model pipeline object.
    """

    # validate normalization type:
    if norm_type != "standard":
        raise Exception("Invalid normalization type.")

    normalizer = None
    # create sklearn normalization object:
    if norm_type == "standard":
        normalizer = StandardScaler

    # create pipeline:
    pipe = Pipeline(steps=[
        ("normalizer", normalizer),
        ("model", model)
    ])

    return pipe

