# Goal - predict the opening stock price 


# Part 1 - Data Preprocessing

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# create a dataframe and then create a numpy array
training_set = dataset_train.iloc[:, 1:2].values


# standardization vs normalization of feature scaling - what why choose
# rnn uses normalization - recommended to use normalization with sigmoid activation
# function in RNNs

# feature scaling
from sklearn.preprocessing import MinMaxScaler
# initialize a MinMaxScaler with a feature range of 0-1 for scaled featurs
sc = MinMaxScaler(feature_range = (0,1))
# apply sc object on data to apply normalization
# fit means its going to get the min and max to apply the normalization formula
# transform means to trasnform each stock price according to the formula
training_set_scaled = sc.fit_transform(training_set)

# create a data structure with 60 timesteps and 1 output

#at each time, the algo is going to look at 59 previous prices and the price we are at right now, 
# and then predict the next ouput at the one we are at rn. the rnn will learn and understand relationships and trends
# to predict the next output, at time
# a smaller timestep leads to overfitting, 20 is too small to caputure trends, but more than 60 is 
# worse as well. 
# the 60 refers to 60 previous financial days

# create the X_train for input values of the training data, feed into model
# y_train will have output of the training data
# at each financial day, x_train will have 60 previous stock prices and y train will contain stock
# price for the next financial day

X_train = []
y_train = []

# use a for loop for every stock price
# from first valid index to the last index. 
for i in range(60, 1258):
    # the rnn learns from 60 previous days in shrot term memory to understand next which is at i
    X_train.append(training_set_scaled[i - 60: i, 0])
    y_train.append(training_set_scaled[i, 0])
# transform lists to np arrays to be able to feed into rnn 
X_train, y_train = np.array(X_train), np.array(y_train)

# now add a new dimension to the data structure that will help predict trends 
# reshaping - use to add a new dimension to the array, the dimension has more indicators
# that can be used for prediction
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# building the neural network

# importing libraries
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# initialize the RNN
regressor = Sequential()

# add the first lstm layer and add dropout regularization to avoid overfitting
# add LSTM object to regressor
# arguments
# number of units/neurons  - want high dimensionality by adding large neurons to capture trends
# so we need high number of neurons
# return sequences = True, build an LSTM that returns sequences to add more in the future
# input shape - shape of the input, in our case its 2D because the first observations will
# automatically be taken into accunt, we only specify our two timesteps and indicators
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))

# add droput - standard amount of 20% neurons will be ignored during forward prop and back prop
regressor.add(Dropout(0.2))


# adding a second LSTM layer and some Dropout regularization
# we don't need to specify shape for second layers and so on it will be recognized

regressor.add(LSTM(units = 50, return_sequences = True))
# add droput - standard amount of 20% neurons will be ignored during forward prop and back prop
regressor.add(Dropout(0.2))


# adding a third LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))
# add droput - standard amount of 20% neurons will be ignored during forward prop and back prop
regressor.add(Dropout(0.2))

# adding a fourth LSTM layer and some Droput regularisation
# last layer has no return sequences
regressor.add(LSTM(units = 50, return_sequences = False))
# add droput - standard amount of 20% neurons will be ignored during forward prop and back prop
regressor.add(Dropout(0.2))

# adding the output layer of dense class
# we need one neuron for the output
regressor.add(Dense(units = 1))


# compile the RNN with an optimzer and the right loss, than fit the NN to the training set

# compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting the rnn to the training set
# 100 is needed for some convergence
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making predictions and visualizing results


# getting the real stock price of 2017

# import the training set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# create a dataframe and then create a numpy array
real_stock_price = dataset_test.iloc[:, 1:2].values



# getting the predicted stock price of 2017
# first input is what we want to concatenate, whcih is google stock prices from 2012 - 2016, and 2017 on
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# get the inputs for each financial day t, get the 60 previous days stock prices
# gets all the inputs we need to predict the value using the model
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# reshape the input with all the data in one column
inputs = inputs.reshape(-1, 1)
# get the right 3d format for to put into predict function to get predictions
# scale the inputs because right now we have the og values of data
inputs = sc.transform(inputs)

# make the special structure - change lower bound because test set is smaller
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
# reshape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
# inverse the scaling of the stock price so we can get the value in real terms
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# visualizing the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()























