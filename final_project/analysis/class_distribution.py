"""Script for plotting class distribution."""


import numpy as np
import matplotlib.pyplot as plt
from final_project.preprocessing.load_data_class import DataLoader
from final_project.preprocessing.preprocess_class import Preprocessor


# data file name:
data_file = "../data/student_performance_train.csv"
# mission type:
mission = 3
# names of binary/nominal features:
bin_feature_names = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities",
                     "nursery", "higher", "internet", "romantic"]
nom_feature_names = ["Mjob", "Fjob", "reason", "guardian"]
# bins for quantizing labels:
bins = np.array([20.5, 16, 14, 12, 10, 0])


# load data:
loader = DataLoader()
X_orig, y_orig = loader.load_and_split_data(data_file, mission)
# preprocess data:
prep = Preprocessor()
X, y = prep.preprocess_data(X_orig, y_orig, bin_feature_names, nom_feature_names, bins, train=True)

# get class distribution:
class_vals, class_counts = np.unique(y, return_counts=True)
# normalize distribution to sum to 1:
class_freq = class_counts / np.sum(class_counts)
# generate class labels:
class_labels = []
for k in class_vals:
    class_labels.append("Class " + str(k+1))

# plot class distribution as bar graph:
fig, ax = plt.subplots()
ax.bar(bins[1:], class_freq, align="center")
ax.set_xticks(bins[1:], class_labels)
ax.set_xlabel("Class Cut-Off Grade")
ax.set_ylabel("Class Frequencies")
ax.set_title("Class Distribution")


# show plots:
plt.show()

