# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing libraries
import pandas as pd

#import datasets
movie_title_df = pd.read_csv('Movie_Id_Titles')

# u.data contains data separated by tab i.e "\t" and name of the columns are = ['user_id','item_id','rating','timestamp']
movie_ratings_df = pd.read_csv('u.data', sep = "\t", names = ['user_id','item_id','rating','timestamp'])
#drop timestamp column
movie_ratings_df = movie_ratings_df.drop('timestamp', axis=1)

movie_description = movie_ratings_df.describe()
movie_ratings_df.info()

movies_rating_df = pd.merge(movie_ratings_df, movie_title_df, on="item_id")

userid_movietitle_matrix = movies_rating_df.pivot_table(index='user_id', columns='title', values='rating')

#Recommender system initiation
movie_correlations = userid_movietitle_matrix.corr(method='pearson', min_periods=80)
myRatings = pd.read_csv('My_Ratings.csv')

similar_movies_list = pd.Series()

for i in range(0,2):
    similar_movies = movie_correlations[myRatings['Movie Name'][i]].dropna()
    similar_movies = similar_movies.map(lambda x:x*myRatings['Ratings'][i])
    similar_movies_list = similar_movies_list.append(similar_movies)
    
similar_movies_list.sort_values(inplace = True, ascending = False)




