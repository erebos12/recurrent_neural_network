import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# Create a numpy array of one column
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling (Standardisation / Normalisation)
# we use Normalisation because we use a sigmoid activation function
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
#   - use t-60 days (3 previous months) values to predict values (past information from RNN will learn)
#   - trying to figure out for timesteps you need to experiment with this values
#   - RNN memorizes the 60 previous stock prices to predict the new value
X_train = []
y_train = []
# get the 60 previous stock prices
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the data by adding a new dimension
#   - adding more indicators, so more dimensions
#   - input dimension represents the input shape expected by RNN
# here three dimenbsion:
#   1. number of stock prices
#   2. number of timesteps
#   3. number of indicators

number_of_rows = X_train.shape[0]
number_of_columns = X_train.shape[1]
X_train = np.reshape(X_train, (number_of_rows, number_of_columns, 1))

print('all fine')
