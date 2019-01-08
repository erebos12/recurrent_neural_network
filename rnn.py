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

############################
# Building the RNN
############################
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
#  - units = number of input neurons
#  - return_sequences=,
#      input_shape=)
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Adding Dropout layer
#    - 20% of neurons will be ignored in LSTM in each training iteration
#    - so 20% of 50 neurons is 10, so each training 10 neurons will be ignored/dropped
regressor.add(Dropout(rate=0.2))
# Adding extra LSTM layers

# add second LSTM layer and Dropout regularisation but no input_shape
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# add third LSTM layer and Dropout regularisation but no input_shape
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# add fourth LSTM layer and Dropout regularisation but no input_shape
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# Adding output layer
regressor.add(Dense(units=1))

# Compiling RNN
#  - good optimizer for RNNs are RMSprop or Adam
#  - regression problem should use mean-squared-error as loss function
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Train RNN with Training set
#  - updating the weights every batch_size=32/64 ...
#  - epochs 100 is better (30 just because of duration reduction for training)
regressor.fit(X_train, y_train, epochs=30, batch_size=32)

#####################################
# Making prediction and visualize it
# --> Predict Google stock price for Jan 2017
#####################################

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017

column_name = 'Open'
vertical_axis = 0
# Getting dataframe from Open-Google-Stock-Prices from 2012, 2016 and 2017
#  - concatenation of test and train set but we need to scale the sets, but test set must be untouched
#  - scaling because RNN was trained by scaled/normalized values
dataset_total = pd.concat((dataset_train[column_name], dataset_test[column_name]), axis=vertical_axis)
lower_bound = len(dataset_total) - len(dataset_test) - 60
inputs = dataset_total[lower_bound:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# Getting the 3D format
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predict stock price
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price in JAN 2017')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price in JAN 2017')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

print('DONE')
