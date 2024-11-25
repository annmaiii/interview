


#import libraries
import os
import datetime as dt
import logging
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# # Generate sample_transformed dataset for quantative reports:



# Preprocess Data Function
def preprocess_data(input_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Remove duplicate rows based on the 'description' column
    df.drop_duplicates(subset='description', keep='first', inplace=True)

    # Handle missing values
    fill_columns = ['director', 'cast', 'country', 'rating', 'duration']
    df[fill_columns] = df[fill_columns].fillna('Unknown')

    # Date processing
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['added_year'] = df['date_added'].dt.year
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['age_to_netflix'] = df['added_year'] - df['release_year']

    # Clean text columns
    def clean_text(column):
        return column.str.replace(r'[^A-Za-z0-9\s]', '', regex=True).str.replace(r'\n', ' ', regex=True)

    for col in ['title', 'description']:
        df[col] = clean_text(df[col])

    # Process 'duration'
    df['minutes'] = df['duration'].apply(lambda x: int(x.split(' ')[0]) if 'min' in x else None)
    df['seasons'] = df['duration'].apply(lambda x: int(x.split(' ')[0]) if 'Season' in x else None)

    # Generalized bucketing function
    def bucketize(value, buckets):
        for condition, label in buckets:
            if condition(value):
                return label
        return None

    # Buckets for minutes and seasons
    minute_buckets = [
        (pd.isna, None),
        (lambda x: x <= 60, 'Short (<1 hr)'),
        (lambda x: 61 <= x <= 120, 'Medium (1-2 hrs)'),
        (lambda x: x > 120, 'Long (>2 hrs)'),
    ]
    df['minute_buckets'] = df['minutes'].apply(lambda x: bucketize(x, minute_buckets))

    season_buckets = [
        (pd.isna, None),
        (lambda x: x == 1, '1 Season'),
        (lambda x: 2 <= x <= 3, '2-3 Seasons'),
        (lambda x: x > 3, '4+ Seasons'),
    ]
    df['season_buckets'] = df['seasons'].apply(lambda x: bucketize(x, season_buckets))

    # Group ratings
    def group_ratings(rating):
        kids = ['TV-Y', 'TV-Y7', 'G', 'TV-G', 'TV-PG', 'TV-Y7-FV']
        teens = ['PG', 'PG-13', 'TV-14']
        mature = ['R', 'NC-17', 'TV-MA']
        if rating in kids:
            return 'Kids/Family Friendly'
        elif rating in teens:
            return 'Teen/Young Adult'
        elif rating in mature:
            return 'Mature Audiences'
        return 'Not Rated/Unknown'

    df['rating_group'] = df['rating'].apply(group_ratings)

    # Cast size
    df['cast_size'] = df['cast'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

    # Function to explode specified columns
    def explode_columns(df, columns):
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna('').str.split(', ')
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x])
                df = df.explode(col, ignore_index=True)  # Explode and reset index
                df[col] = df[col].str.replace(',', '').str.strip()  # Remove commas and strip spaces

        return df

    # Explode multi-valued columns
    columns_to_explode = ['director', 'cast', 'country', 'listed_in']
    df = explode_columns(df, columns_to_explode)

    # Select relevant columns
    relevant_columns = [
        'show_id','type','title', 'director', 'cast', 'cast_size','country', 'rating_group','rating',
        'added_year', 'date_added','release_year', 'age_to_netflix', 'minute_buckets',
        'season_buckets', 'listed_in'
    ]
    df = df[relevant_columns]
    return df


def upload_data(output_path):
    df_transformed.to_csv(output_path, index = False)



input_path = 'sample.csv'
df_transformed = preprocess_data(input_path)

output_path =  'sample_transformed.csv'
output = upload_data(output_path)
print(f"Transforming completed. Output saved to '{output_path}'.")


# # Generate sample_processed_clustered dataset for text report


# Preprocess Data Function
def preprocess_data(input_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Remove duplicate rows based on the 'description' column
    df.drop_duplicates(subset='description', keep='first', inplace=True)

    # Handle missing values
    fill_columns = ['director', 'cast', 'country', 'rating', 'duration']
    df[fill_columns] = df[fill_columns].fillna('Unknown')

    # Date processing
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['added_year'] = df['date_added'].dt.year
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df['age_to_netflix'] = df['added_year'] - df['release_year']

    # Clean text columns
    text_columns = ['title', 'description']
    for col in text_columns:
        df[col] = df[col].fillna('').str.replace(r'[^A-Za-z0-9\s]', '', regex=True).str.replace(r'\n', ' ', regex=True)

    # Group ratings
    rating_map = {
        **dict.fromkeys(['TV-Y', 'TV-Y7', 'G', 'TV-G', 'TV-PG', 'TV-Y7-FV'], 'Kids/Family Friendly'),
        **dict.fromkeys(['PG', 'PG-13', 'TV-14'], 'Teen/Young Adult'),
        **dict.fromkeys(['R', 'NC-17', 'TV-MA'], 'Mature Audiences')
    }
    df['rating_group'] = df['rating'].map(rating_map).fillna('Not Rated/Unknown')

    return df


# Text Preprocessing for Clustering
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()  # Remove special characters and convert to lowercase
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Clustering Function
def cluster_content(df):
    # Combine textual columns for clustering
    df['processed_text'] = (df['title'] + ' ' + df['description']).apply(preprocess_text)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['content_cluster'] = kmeans.fit_predict(tfidf_matrix).astype(str)

    # Select relevant columns
    relevant_columns = [
        'show_id', 'type', 'rating_group', 'date_added',
        'release_year', 'age_to_netflix', 'processed_text', 'content_cluster'
    ]
    df = df[relevant_columns]

    return df


# Main Execution
if __name__ == "__main__":
    input_path = 'sample.csv'  # Path to your dataset

    # Preprocess and cluster content
    df_processed = preprocess_data(input_path)
    df_clustered = cluster_content(df_processed)

    # Save results
    output_file = 'sample_processed_clustered.csv'
    df_clustered.to_csv(output_file, index=False)
    print(f"Preprocessing and clustering completed. Output saved to '{output_file}'.")




