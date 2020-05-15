# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:00:41 2020

@author: drv_m
"""
#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Import Dataset
movie_titles_df = pd.read_csv('Movie_Id_Titles')
movies_rating_df = pd.read_csv('u.data', sep = "\t", names = ["user_id","item_id","rating","timestamp"])
movies_rating_df.drop(['timestamp'], axis = 1, inplace = True )

#Get statistical inferences
movie_rating_df_stats = movies_rating_df.describe()
movies_rating_df.info()

#Merge dataframes to get actual name of movies
movies_rating_df = pd.merge(movies_rating_df, movie_titles_df, on = 'item_id')

#Visualization of dataset
individual_movie_rating_stats = movies_rating_df.groupby('title')['rating'].describe()

ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean']
ratings_df_count = movies_rating_df.groupby('title')['rating'].describe()['count']
ratings_mean_count_df = pd.concat([ratings_df_count, ratings_df_mean], axis = 1)
ratings_mean_count_df = ratings_mean_count_df.reset_index() 


#Plot histogram
ratings_mean_count_df['mean'].plot(bins = 100, kind = 'hist', color = 'r')
ratings_mean_count_df['count'].plot(bins = 100, kind = 'hist', color = 'r')

ratings_mean_count_df[ ratings_mean_count_df['mean'] == 5 ]  












