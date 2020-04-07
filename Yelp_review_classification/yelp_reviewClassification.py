# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:46:17 2020

@author: drv_m
"""
'''Import Libraries'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#NLP libraries
import string
from nltk.corpus import stopwords
stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

'''Import dataset'''
yelp_df = pd.read_csv('yelp.csv')
yelp_df.shape

yelp_df.describe()#gives statistical summary of our DF
yelp_df.info()

#read a review
yelp_df['text'][0]

'''Visualize datase'''
yelp_df['length'] = yelp_df['text'].apply(len)

yelp_df['length'].plot(bins = 100, kind = 'hist')
yelp_df.length.describe()#gives text length statistics from text column
sns.countplot(y = 'stars', data = yelp_df)

#see length of data for each star rating
g = sns.FacetGrid(data = yelp_df, col = 'stars', col_wrap = 5)
g.map(plt.hist, 'length', bins = 20, color = 'b')

#divide review in two groups(1 star and 5 star)
yelp_df_1 = yelp_df[yelp_df['stars'] == 1]
yelp_df_5 = yelp_df[yelp_df['stars'] == 5]

yelp_df_1_5 = pd.concat([yelp_df_1, yelp_df_5])

#Check if dataset is balanced
sns.countplot(yelp_df_1_5['stars'], label = 'Count')

'''Remove stopwords'''
def message_cleaning(message):
    Test_punc_removed = [ char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)
print(yelp_df_clean[0])# cleaned up review
print(yelp_df_1_5['text'][0])# original review

'''Apply vectorizer to yelp review'''
vectorizer = CountVectorizer(analyzer = message_cleaning)
yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])
print(vectorizer.get_feature_names())

print(yelp_countvectorizer.toarray())

'''Model training'''

NB_classifier = MultinomialNB()
label = yelp_df_1_5['stars'].values#target class
NB_classifier.fit(yelp_countvectorizer, label)#input=yelp_countvectorizer, output=label

testing_sample = ['amazing food! highly recommended']
#testing_sample = ['shit food, made me sick']

#transform text --> numbers using vectorizer
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)

X = yelp_countvectorizer
y = label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
NB_classifier.fit(X_train, y_train)

'''Evaluating model'''
y_predict_train = NB_classifier.predict(X_train)
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True)

y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_predict_test))








