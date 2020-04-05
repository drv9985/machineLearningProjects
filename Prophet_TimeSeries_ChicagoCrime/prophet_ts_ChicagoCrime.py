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
import matplotlib.pyplot as plt


'''Import datasets'''
chicago_df_1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', error_bad_lines = False)
chicago_df_2 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', error_bad_lines = False)
chicago_df_3 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', error_bad_lines = False)

'''Exploratory Data Analysis'''
chicago_df_1.shape
chicago_df_2.shape
chicago_df_3.shape

chicago_df = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3])
chicago_df.shape
columns_list = list(chicago_df.columns)

#drop columns that are not needed
chicago_df.drop(['Unnamed: 0', 'Case Number', 'ID', 'IUCR', 'X Coordinate', 'Y Coordinate', 'Updated On', 'Year', 'FBI Code', 'Beat', 'Ward', 'Community Area', 'Location', 'District', 'Latitude', 'Longitude'], inplace=True, axis=1)

#convert Date column to date-time format
chicago_df.Date = pd.to_datetime(chicago_df.Date, format = '%m/%d/%Y %I:%M:%S %p')
chicago_df.Date
#setting up date column in the dataframe as index of dataframe
chicago_df.index = pd.DatetimeIndex(chicago_df.Date)

crime_type = chicago_df['Primary Type'].value_counts()

#considering top-15 Primary_type crime
crime_type_data = chicago_df['Primary Type'].value_counts().iloc[:15].index
location_data = chicago_df['Location Description'].value_counts().iloc[:15].index
#crime type-primary type visualization
sns.countplot(y = 'Primary Type', data = chicago_df, order = crime_type_data)
#crime type-location description visualization
sns.countplot(y = 'Location Description', data = chicago_df, order = location_data)
#Get occurence of crime by date
#Note: resample function is used to find out frequency based on month(M), year(Y), quarter(Q), etc.
chicago_df.resample('Y').size()
#resample(Year) visualization
plt.plot(chicago_df.resample('Y').size())
plt.title('Crime Count Per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
#resample(Month) visualization
plt.plot(chicago_df.resample('M').size())
plt.title('Crime Count Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
#resample(Quarter) visualization
plt.plot(chicago_df.resample('Q').size())
plt.title('Crime Count Per Quarter')
plt.xlabel('Quarter')
plt.ylabel('Number of Crimes')

'''Preparing the data'''
chicago_prophet = chicago_df.resample('M').size().reset_index()
chicago_prophet.columns = ['Date', 'Crime Count']
chicago_prophet

#facebook prophet needs columns to be renamed as ds and y
chicago_prophet_df_final = chicago_prophet.rename(columns = {'Date' : 'ds', 'Crime Count' : 'y'})

'''Make predictions with FB prophet'''
m = Prophet()
m.fit(chicago_prophet_df_final)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
#visualization of forecast
figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Crime Rate')
#visualiztion of seasonality
figure = m.plot_components(forecast)





