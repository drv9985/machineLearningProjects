# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:09:10 2020

@author: drv_m
Title: Prophet Time Series forecasting of Chicago Crime
"""

'''Import Libraries'''
import numpy as np
import pandas as pd
import seaborn as sns
from fbprophet import Prophet


'''Import datasets'''
chicago_df_1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', error_bad_lines = False)
chicago_df_2 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', error_bad_lines = False)
chicago_df_3 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', error_bad_lines = False)