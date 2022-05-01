"""Script for plotting class distribution."""


import numpy as np
import matplotlib.pyplot as plt
from final_project.preprocessing.load_data_class import DataLoader
from final_project.preprocessing.preprocess_class import Preprocessor


# data file name:
data_file = "../data/student_performance_train.csv"
# mission type:
mission = 3
# bins for quantizing labels:
bins_quantize = np.array([20.5, 16, 14, 12, 10, 0])


# load data:
loader = DataLoader()
X_orig, y_orig = loader.load_and_split_data(data_file, mission)
# preprocess data:
prep = Preprocessor()
X, y = prep.preprocess_data(X_orig, y_orig, bins=bins_quantize, train=True)

# plot class prior probabilities as a bar graph:
class_priors = prep.class_priors
# generate class labels:
class_labels = []
for k in range(len(class_priors)):
    class_labels.append("Class " + str(k+1))
# make plot:
fig, ax = plt.subplots()
ax.bar(bins_quantize[1:], class_priors, align="center")
ax.set_xticks(bins_quantize[1:], class_labels)
ax.set_xlabel("Class Cut-Off Grade")
ax.set_ylabel("Class Priors")
ax.set_title("Class Prior Probabilities")


# show plots:
plt.show()

