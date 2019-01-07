import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# Create a numpy array of one column
training_set = dataset_train.iloc[:, 1:2].values
print(training_set)

# Feature Scaling (Standardisation / Normalisation)
# we use Normalisation because we use a sigmoid activation function
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Building the RNN

