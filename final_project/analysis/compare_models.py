"""Script for comparing cross-validation performance of all models."""


import numpy as np
import matplotlib.pyplot as plt


# dictionaries mapping mission, mode, and metric keys to their full names:
mission_names = {
    "mission1": "Mission 1",
    "mission2": "Mission 2",
    "mission3": "Mission 3"
}
model_names = {
    "trivial": "Trivial",
    "baseline": "Baseline",
    "knn": "KNN",
    "logistic": "Logistic",
    "svm": "SVM",
    "nearest_means": "Nearest Means"
}
metric_names = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1-Score"
}


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
    },
    "nearest_means": {
        "accuracy": 0.309,
        "macro_f1": 0.285
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
    },
    "nearest_means": {
        "accuracy": 0.364,
        "macro_f1": 0.358
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
        "accuracy": 0.661,
        "macro_f1": 0.647
    },
    "svm": {
        "accuracy": 0.765,
        "macro_f1": 0.754
    },
    "nearest_means": {
        "accuracy": 0.755,
        "macro_f1": 0.748
    }
}

# combine all missions into a single dictionary for convenience:
all_metrics = {
    "mission1": mission1_metrics,
    "mission2": mission2_metrics,
    "mission3": mission3_metrics
}


print()
# nested dictionaries with missions as 1st-level keys and metric names as 2nd-level keys:
best_models = {}
best_metrics = {}
# determine best-performing models, separately for each mission:
for mission in all_metrics.keys():
    # cross-validation metrics for current mission:
    metrics_mission = all_metrics[mission]
    metric_types = list(metrics_mission[list(metrics_mission.keys())[0]].keys())

    # dictionaries with metric names as keys:
    best_models_mission = {}
    best_metrics_mission = {}
    for metric_type in metric_types:
        best_metrics_mission[metric_type] = 0.0

    # find best-performing models, with respect to different metrics:
    for model in metrics_mission.keys():
        for metric_type in metric_types:
            metric_value = metrics_mission[model][metric_type]
            if metric_value > best_metrics_mission[metric_type]:
                best_models_mission[metric_type] = model
                best_metrics_mission[metric_type] = metric_value
    best_models[mission] = best_models_mission
    best_metrics[mission] = best_metrics_mission

# print best-performing models:
for mission in all_metrics.keys():
    print()
    print("For " + mission_names[mission] + ":")
    metric_types = list(all_metrics[mission][list(all_metrics[mission].keys())[0]].keys())
    for metric_type in metric_types:
        print("Best-performing model w.r.t. " + metric_type + ": " + model_names[best_models[mission][metric_type]])


# plot bar graphs of cross-validation performance to compare models:
bar_width = 0.25
fig_num = 1
for mission in all_metrics.keys():
    # cross-validation metrics for current mission:
    metrics_mission = all_metrics[mission]
    n_models = len(list(metrics_mission.keys()))

    # get list of model keys and their full names, in same order as the lists in the metrics dictionary below:
    models = []
    model_names_plot = []
    for model in metrics_mission.keys():
        models.append(model)
        model_names_plot.append(model_names[model])
    # build dictionary of metrics, where metrics[metric_name] = list of metric_name values for all models:
    metrics = {}
    metric_types = list(metrics_mission[list(metrics_mission.keys())[0]].keys())
    for metric_type in metric_types:
        metric_values = []
        for model in metrics_mission.keys():
            metric_values.append(metrics_mission[model][metric_type])
        metrics[metric_type] = metric_values

    # setup positions of bars:
    bar_pos = {}
    k = 0
    for metric_type in metric_types:
        bar_pos[metric_type] = np.arange(n_models) + k * bar_width
        k += 1

    # plot all metrics on same plot:
    fig = plt.subplots()
    for metric_type in metric_types:
        plt.bar(bar_pos[metric_type], metrics[metric_type], width=bar_width, align="edge",
                label=metric_names[metric_type])
    # label bars by model (more accurately, groups of bars):
    plt.xticks(np.arange(n_models) + 0.5*len(metrics)*bar_width, model_names_plot)
    # annotate plot:
    plt.title(mission_names[mission] + " Comparison of Models")
    plt.ylabel("Cross-Validation Performance Value")
    plt.legend()


# show plots:
plt.show()

