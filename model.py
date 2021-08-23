# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


# import data
dataset = pd.read_csv("hiring.csv")

# fill experience null values with 0
dataset["experience"].fillna(0, inplace=True)

# fill test score null value with mean
dataset["test_score"].fillna(dataset["test_score"].mean(), inplace=True)

# Cretae input feature
X = dataset.iloc[:, :3]

# Converting categories to integer values
def convert_to_numeric(word):
    word_dict = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "zero": 0,
        0: 0,
    }
    return word_dict[word]


X["experience"] = X["experience"].apply(lambda x: convert_to_numeric(x))

# target value
y = dataset.iloc[:, -1]


# Linear Regression model
regressor = LinearRegression()

# Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open("model.pkl", "wb"))
