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