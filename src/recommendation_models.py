"""
Recommendation Algorithm for Spotify Song data.

Author: RJ
Date: 12/09/2023

Model 1: K-Means Clustering
Model 2: Content-Based Filtering
"""


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


def recommend_songs_cosine_similarity(song_name, data, similarity_matrix, top_n=10):
    """
    Recommend top n songs closest to the seed song based on cosine similarity.

    Args:
        song_name: Name of the song to base recommendations on.
        data: Pandas DataFrame of song data.
        similarity_matrix: Numpy array of cosine similarity scores.
        top_n: Number of top recommendations to return.

    Returns:
        Pandas DataFrame of top n recommended songs closest to the seed song.
    """
    if song_name not in data['name'].values:
        return "Song not found in the dataset."

    # Find the index of the seed song
    song_index = data[data['name'] == song_name].index[0]

    # Get the similarity scores of the seed song with all other songs
    song_similarities = similarity_matrix[song_index]

    # Create a DataFrame of songs and their similarity scores
    recommendations = pd.DataFrame({'name': data['name'], 'artists': data['artists'], 'year': data['year'], 'similarity': song_similarities})

    # Sort the songs based on similarity scores
    recommendations = recommendations.sort_values(by='similarity', ascending=False)

    # Exclude the seed song and return the top n songs
    recommendations = recommendations[recommendations['name'] != song_name]
    return recommendations[['name', 'artists', 'year']].head(top_n)
    
    