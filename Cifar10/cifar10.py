# -*- coding: utf-8 -*-
"""
Spyder Editor

author: drv_muk
"""

'''Import libraries'''
# Conv2D: for convolution process
# MaxPooling2D: get max pixel, for downn sampling/compress features
# AveragePooling2D: get average pixel, for downn sampling/compress features
# Dense: to create fully connected Atificial neural network
# Flatten: Flatten features to one array of neuron
# Dropout: to perform some regularization
# Adam: to perform optimization, to get weight of the network

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10#load CIFAR 10 dataset
import keras#used for normalization
from keras.models import Sequential#for model building in sequential fashion(Left to Right)
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
''''''

'''Exploratory Data Aanalysis'''
#Get shapes
X_train.shape
X_test.shape
y_train.shape
y_test.shape

#Visualization
i = 1000
plt.imshow(X_train[i])
print(y_train[i])

#Visualize some random images
W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training) # pick a random number
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])#show label of the images
    axes[i].axis('off') # get rid of x and y axis in image
    
plt.subplots_adjust(hspace = 0.4)

'''Data Preparation'''

# converting image to float 32 for DL algorithm consumption. 
# Float is chosen for normalization, so that we can havedecimal numbers.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')    

number_categories = 10 # since there are 10 categories in CIFAR 10

# change decimal numbers i.e 0,1,2,3,4...9 to categorical ~ 10 numbers
y_train = keras.utils.to_categorical(y_train, number_categories)
y_test = keras.utils.to_categorical(y_test, number_categories)

X_train = X_train/255# performed normalization to get numbers from 0-1
X_test = X_test/255

Input_shape = X_train.shape[1:]# get input shape of image i.e 32x32

'''Create the model'''
#Activation function:
# 1.Relu = creates output that is continuous
# 2.Softmax = will give output as 1/0, we will use this at the output layer because it is a classification problem

# Build CNN model in sequential form
cnn_model = Sequential()# create sequential object
# I am choosing 32 filters(it is like instagram filters~blur,sharpen etc) and the filter size will be of 3x3 matrix
cnn_model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape=(Input_shape)))#Layer 1
cnn_model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))#Layer 2, donot need to specify input shape here because it will inherit it from 1st layer
# Specify max_pooling of 2x2 matrix for down-sampling
cnn_model.add(MaxPooling2D(2,2))
# Dropout 30% neurons for regularization
cnn_model.add(Dropout(0.3))

# Adding depth to the network
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.2))

# Perform flattening
cnn_model.add(Flatten())#flatten features to single array

# Dense acts as a network/connection/wirining between neurons, units specify number of neurons in hidden layer
cnn_model.add(Dense(units = 512, activation = 'relu'))# Creates hidden layer
cnn_model.add(Dense(units = 512, activation = 'relu'))

#Output layer: units is 10 because we have 10 categories in CIFAR-10, Used softmax because classification problem
cnn_model.add(Dense(units = number_categories, activation = 'softmax'))

'''Train the model'''
# Categorical cross entropy: If we use this loss, we will train a CNN to output a probability over the C classes for each image. It is used for multi-class classification.
# Optimizer: Optimizers update the weight parameters to minimize the loss function. Loss function acts as guides to the terrain telling optimizer if it is moving in the right direction to reach the bottom of the valley, the global minimum.
cnn_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.rmsprop(lr = 0.001), metrics = ['accuracy'])

# create a history variable to log all changes done while the model gets trained
# shuffle = True, will shuffle the images
history = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 2, shuffle = True)

'''Evaluate model'''


