# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 08:44:42 2020

@author: drv_m
"""

'''Importing Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from fbprophet import Prophet

'''Import dataset'''
avocado_df = pd.read_csv('avocado.csv')

'''Exploratory Data Analysis'''
avocado_df.shape
column_list = avocado_df.columns
#sort dataframe by date
avocado_df = avocado_df.sort_values("Date")

#plot date vs avg price of Avocado
plt.figure(figsize = (10, 10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])

#plot count of avocado sale per region
plt.figure(figsize = (25,12))
sns.countplot(data = avocado_df, x = 'region')#shows the data is balanced as we have data from many regions
plt.xticks(rotation = 45)

#plot count of avocado sale per year
sns.countplot(x = 'year', data = avocado_df)

#create df for fbprohet consumption
avocado_prophet_df = avocado_df[['Date','AveragePrice']]




