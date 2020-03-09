# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:53:36 2020

@author: drv_muk
"""

'''Step:1-Import Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Step:2-Import Dataset'''
car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')

sns.pairplot(car_df)

'''Step:3-Preparing dataset for DL'''
X = car_df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'], axis=1)
y = car_df['Car Purchase Amount']
X.shape#(500,5)
y.shape#(500,)

#Scaling/Normalization(Min-Max)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

y = y.values.reshape(-1,1)
y.shape#(500,1)
y_scaled = scaler.fit_transform(y)

'''Step:4-Training the model'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

X_train.shape
X_test.shape

import tensorflow.keras
from keras.models import Sequential#build network like a Lego,sequentially
from keras.layers import Dense#build a fully connected network

model = Sequential()
model.add(Dense(25, input_dim = 5, activation = 'relu'))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.summary()#shows summary of our model

model.compile(optimizer = 'adam', loss = 'mean_squared_error')#adam for gradient descent/back-propagation
epochs_hist = model.fit(X_train, y_train, epochs = 20, batch_size = 25, verbose = 1, validation_split = 0.2)

'''Step:5-Evaluating the model'''
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])

#Gender, Age, Annual Salary, Credit Card Debt, Net Worth
X_test = np.array([[1, 50, 50000, 10000, 600000]])
y_predict = model.predict(X_test)
print('Expected Purchase Amount', y_predict)
