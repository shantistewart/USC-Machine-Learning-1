"""Script for initial exploration of data."""


import pandas as pd
from final_project.preprocessing.load_data_class import DataLoader

# configure pandas display options:
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 50)


# data file name:
data_file = "../data/student_performance_train.csv"


print("")
# load data:
get_data = DataLoader()
data = get_data.load_data(data_file)

# shape of dataframe:
print("Shape of data frame: {}\n".format(data.shape))
# display column names:
print(data.columns)
# display basic info:
print()
print(data.info())
# display basic statistics:
print()
print(data.describe())
# preview data:
print()
print(data.head())

# check for missing data:
print()
print(data.isnull().sum())

