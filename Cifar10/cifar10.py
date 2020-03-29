# -*- coding: utf-8 -*-
"""
Spyder Editor

author: drv_muk
"""

'''Import libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
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
  
    
    
    
    
    