# -*- coding: utf-8 -*-
"""Book Recommendation System.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZxnN8EmWwpBONadAtA8FK1VUnpzToV_H
"""

import numpy as np
import pandas as pd
import seaborn as sns

books = pd.read_csv('OneDrive/Desktop/Book Recommender system/Books.csv')

ratings=pd.read_csv('Ratings.csv')
users=pd.read_csv('Users.csv')

books.head()

ratings.head()

users.head()

print(books.shape)
print(ratings.shape)
print(users.shape)

books.isnull().sum()

users.isnull().sum()

ratings.isnull().sum()

books.duplicated().sum()
ratings.duplicated().sum()
users.duplicated().sum()

## Popolarity based recommender system
ratings_with_name=ratings.merge(books,on='ISBN')

num_rating_df=ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num-rating'},inplace=True)
num_rating_df

avg_rating_df=ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg-rating'},inplace=True)
avg_rating_df

# Filter popular books (e.g. 250+ ratings) and sort by average rating
popular_df = popularity_df[popularity_df['num-rating'] >= 250].sort_values('avg-rating', ascending=False)
popular_df

# merge popular books with again books col
popularity_df=popularity_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num-rating','avg-rating']]

popularity_df

## Collaborative based filtering, index of Users who gave rating to more than 200 books
x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
padhe_likhe_users=x[x].index

# Filtering done on the basis of users
filtered_ratings=ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

# Now on the basis of books
y=filtered_ratings.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index

final_ratings=filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]

final_ratings.drop_duplicates()

pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')

pt.fillna(0,inplace=True)

pt

from sklearn.metrics.pairwise import cosine_similarity
similarity_scores=cosine_similarity(pt)

def recommend(book_name):
  index=np.where(pt.index==book_name)[0][0]
  distances=similarity_scores[index]
  similar_items=sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
  for i in similar_items:
    print(pt.index[i[0]])

recommend('The Notebook')

pt.index

sorted(list(enumerate(similarity_scores[0])),key=lambda x:x[1],reverse=True)[1:6]
