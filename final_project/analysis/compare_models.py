"""Script for comparing performance of models."""


import numpy as np
import matplotlib.pyplot as plt

# mission 1 cross-validation performance metrics:
mission1_metrics = {
    "trivial": {
        "accuracy": 0.233,
        "macro_f1": 0.201
    },
    "baseline": {
        "accuracy": 0.344,
        "macro_f1": 0.311
    },
    "knn": {
        "accuracy": 0.325,
        "macro_f1": 0.281
    },
    "logistic": {
        "accuracy": 0.362,
        "macro_f1": 0.303
    },
    "svm": {
        "accuracy": 0.385,
        "macro_f1": 0.317
    }
}

# mission 2 cross-validation performance metrics:
mission2_metrics = {
    "trivial": {
        "accuracy": 0.216,
        "macro_f1": 0.188
    },
    "baseline": {
        "accuracy": 0.342,
        "macro_f1": 0.334
    },
    "knn": {
        "accuracy": 0.376,
        "macro_f1": 0.348
    },
    "logistic": {
        "accuracy": 0.331,
        "macro_f1": 0.289
    },
    "svm": {
        "accuracy": 0.379,
        "macro_f1": 0.342
    }
}

# mission 3 cross-validation performance metrics:
mission3_metrics = {
    "trivial": {
        "accuracy": 0.220,
        "macro_f1": 0.193
    },
    "baseline": {
        "accuracy": 0.486,
        "macro_f1": 0.490
    },
    "knn": {
        "accuracy": 0.765,
        "macro_f1": 0.754
    },
    "logistic": {
        "accuracy": 0,
        "macro_f1": 0
    },
    "svm": {
        "accuracy": 0.765,
        "macro_f1": 0.754
    }
}

# combine all missions into a single dictionary for convenience:
metrics = {
    "mission1": mission1_metrics,
    "mission2": mission2_metrics,
    "mission3": mission3_metrics
}

