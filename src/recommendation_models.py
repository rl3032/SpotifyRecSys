"""
Recommendation Algorithm for Spotify Song data.

Author: RJ
Date: 12/09/2023

Model 1: K-Means Clustering
Model 2: Content-Based Filtering
"""

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def recommend_songs_keans(song_name, data, kmeans_model, scaler, features, top_n=10):
    """
    Recommend top n songs closest to the seed song based on KMeans clustering.

    Args:
        song_name: Name of the song to base recommendations on.
        data: Pandas DataFrame of song data.
        kmeans_model: Trained KMeans model.
        scaler: StandardScaler object used for feature scaling.
        features: List of features used in clustering.
        top_n: Number of top recommendations to return.

    Returns:
        Pandas DataFrame of top n recommended songs closest to the seed song.
    """
    if song_name not in data['name'].values:
        return "Song not found in the dataset."

    # Find the feature vector of the seed song
    seed_song_features = scaler.transform(data[data['name'] == song_name][features])

    # Predict the cluster for the seed song
    song_cluster = kmeans_model.predict(seed_song_features)[0]

    # Filter out the songs from the same cluster
    cluster_songs = data[data['cluster'] == song_cluster]

    # Calculate similarity scores for all songs in the same cluster
    similarities = cosine_similarity(seed_song_features, scaler.transform(cluster_songs[features]))
    cluster_songs['similarity'] = np.squeeze(similarities)

    # Sort the songs based on similarity scores
    recommendations = cluster_songs.sort_values(by='similarity', ascending=False)

    # Exclude the seed song and return the top n songs
    recommendations = recommendations[recommendations['name'] != song_name]
    return recommendations[['name', 'artists', 'year']].head(top_n)


def recommend_songs_content_based(song_name, data, scaler, features, top_n=10, popularity_weight=0.5):
    """
    Recommend top n songs based on a mix of cosine similarity and popularity.

    Args:
        song_name: Name of the seed song to base recommendations on.
        data: Pandas DataFrame of song data.
        scaler: StandardScaler object used for feature scaling.
        features: List of features used in clustering.
        top_n: Number of top recommendations to return.
        popularity_weight: Weight for popularity in the final score (0 to 1).

    Returns:
        Pandas DataFrame of top n recommended songs.
    """
    if song_name not in data['name'].values:
        return "Seed song not found in the dataset."

    # Find the feature vector of the seed song
    song_features = scaler.transform(data[data['name'] == song_name][features])

    # Calculate similarity scores for all songs
    similarities = cosine_similarity(song_features, scaler.transform(data[features]))
    data['similarity'] = np.squeeze(similarities)

    # Normalize popularity
    data['normalized_popularity'] = data['popularity'] / data['popularity'].max()

    # Combine similarity and popularity
    data['score'] = (1 - popularity_weight) * data['similarity'] + popularity_weight * data['normalized_popularity']

    # Sort the songs based on the combined score
    recommendations = data.sort_values(by='score', ascending=False)

    # Exclude the seed song and return the top n songs
    recommendations = recommendations[recommendations['name'] != song_name]
    return recommendations[['name', 'artists', 'year', 'score']].head(top_n)


    
    