# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:53:36 2020

@author: drv_muk
"""

'''Import Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Import Dataset'''
car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')

sns.pairplot(car_df)

'''Preparing dataset for DL'''
X = car_df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'], axis=1)
y = car_df['Car Purchase Amount']