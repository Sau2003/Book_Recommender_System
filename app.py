# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
books = pd.read_csv('OneDrive/Desktop/Book Recommender system/Books.csv')
ratings = pd.read_csv('Ratings.csv')

# Merge
ratings_with_name = ratings.merge(books, on='ISBN')

# Popularity-based
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num-rating'}, inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popularity_df = popularity_df[popularity_df['num-rating'] >= 250]
popularity_df = popularity_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
    ['Book-Title', 'Book-Author', 'Image-URL-M', 'num-rating', 'avg_rating']
]

# Collaborative filtering
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index

filtered_ratings = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]
y = filtered_ratings.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

similarity_scores = cosine_similarity(pt)

# Recommend function
def recommend(book_name):
    if book_name not in pt.index:
        return []
    index = np.where(pt.index == book_name)[0][0]
    distances = similarity_scores[index]
    similar_items = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    recommendations = []
    for i in similar_items:
        book = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        recommendations.append({
            'title': pt.index[i[0]],
            'author': book['Book-Author'].values[0],
            'image': book['Image-URL-M'].values[0]
        })
    return recommendations

# Streamlit App
st.title("📚 Book Recommendation System")
book_list = pt.index.values
selected_book = st.selectbox("Choose a Book", book_list)

if st.button("Recommend"):
    recs = recommend(selected_book)
    if recs:
        for rec in recs:
            st.image(rec['image'], width=120)
            st.write(f"**{rec['title']}** by *{rec['author']}*")
    else:
        st.warning("Sorry! No recommendations found.")
