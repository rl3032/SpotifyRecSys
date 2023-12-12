"""
Model Evaluation

Author: RJ
Date: 12/11/2023

"""
from sklearn.metrics.pairwise import euclidean_distances

def calculate_diversity_score(recommendations, features):
    """
    Calculate the diversity of the recommended songs.

    Args:
        recommendations: DataFrame containing recommended songs and their features.
        features: List of columns to be used for calculating diversity.

    Returns:
        float: Diversity score.
    """
    features = recommendations[features].values
    distances = euclidean_distances(features)
    diversity_score = distances.sum() / (len(recommendations) * (len(recommendations) - 1))
    return diversity_score


def calculate_novelty_score(recommendations, baseline_popularity):
    """
    Calculate the novelty of the recommended songs.

    Args:
        recommendations: DataFrame containing recommended songs and their features.
        features: List of columns to be used for calculating novelty.

    Returns:
        float: Novelty score.
    """
    novelty_scores = recommendations['popularity'].apply(lambda x: abs(x - baseline_popularity))
    average_novelty = novelty_scores.mean()
    return average_novelty


def calculate_accuracy(recommended_songs, test_songs):
    """
    Calculate the accuracy based on the intersection of recommended songs and test songs.

    Args:
        recommended_songs: DataFrame of recommended songs.
        test_songs: DataFrame of test songs.

    Returns:
        float: Accuracy score.
    """
    recommended_song_set = set(recommended_songs['name'])
    test_song_set = set(test_songs['song_name'])
    intersection = recommended_song_set.intersection(test_song_set)
    accuracy = len(intersection) / len(recommended_song_set) if recommended_song_set else 0
    return accuracy