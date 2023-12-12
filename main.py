import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.recommendation_models import recommend_songs_keans, recommend_songs_content_based
from src.model_evaluation import calculate_diversity_score, calculate_novelty_score, calculate_accuracy

def process_seed_songs(seed_songs, data, kmeans_model, scaler, features, test_data_files, top_n=30):
    results = []

    for seed_song, test_data_file in zip(seed_songs, test_data_files):
        print(f"\nProcessing seed song: {seed_song}")

        # Load test data for the seed song
        test_data = pd.read_csv(test_data_file)

        # KMeans clustering recommendations
        kmean_recommendations = recommend_songs_keans(seed_song, data, kmeans_model, scaler, features, top_n)
        kmean_recommendations_data = data[data['name'].isin(kmean_recommendations['name'].values)]

        # Content-based filtering recommendations
        content_based_recommendations = recommend_songs_content_based(seed_song, data, scaler, features, top_n)
        content_based_recommendations_data = data[data['name'].isin(content_based_recommendations['name'].values)]

        # Calculate diversity and novelty scores
        kmean_diversity_score = calculate_diversity_score(kmean_recommendations_data, features)
        content_based_diversity_score = calculate_diversity_score(content_based_recommendations_data, features)
        baseline_popularity = data[data['name'] == seed_song]['popularity'].values[0]
        kmean_novelty_score = calculate_novelty_score(kmean_recommendations_data, baseline_popularity)
        content_based_novelty_score = calculate_novelty_score(content_based_recommendations_data, baseline_popularity)

        # Calculate accuracy
        kmean_accuracy = calculate_accuracy(kmean_recommendations_data, test_data)
        content_based_accuracy = calculate_accuracy(content_based_recommendations_data, test_data)

        # Store results
        results.append({
            'seed_song': seed_song,
            'kmean_diversity_score': kmean_diversity_score,
            'content_based_diversity_score': content_based_diversity_score,
            'kmean_novelty_score': kmean_novelty_score,
            'content_based_novelty_score': content_based_novelty_score,
            'kmean_accuracy': kmean_accuracy,
            'content_based_accuracy': content_based_accuracy
        })

    return results

def main():
    # Load the dataset
    data = pd.read_csv('data/processed/final_data.csv')

    # Define features for clustering
    features = ['popularity', 'duration_ms', 'explicit', 'acousticness', 'danceability', 'energy', 
                'instrumentalness', 'liveness', 'speechiness', 'valence', 'key', 'loudness', 'mode', 'tempo', 'time_signature', 'year']

    # Standardize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    
    # Perform KMeans clustering
    kmeans_model = KMeans(n_clusters=5)
    labels = kmeans_model.fit_predict(data_scaled)

    data['cluster'] = labels

    
    # List of seed songs
    seed_songs = ["Shape of You", "Something Just Like This", "Don't Start Now", "Watermelon Sugar", "Bad Guy"]
    test_data_files = [
        'data/processed/filtered_songs(Shape of You).csv',
        'data/processed/filtered_songs(Something Just Like This).csv',
        'data/processed/filtered_songs(Don\'t Start Now).csv',
        'data/processed/filtered_songs(Watermelon Sugar).csv',
        'data/processed/filtered_songs(Bad Guy).csv'
    ]

    # Process each seed song
    evaluation_results = process_seed_songs(seed_songs, data, kmeans_model, scaler, features, test_data_files)

    # Display results
    for result in evaluation_results:
        print(f"\nResults for seed song: {result['seed_song']}")
        print(f"KMeans Diversity Score: {result['kmean_diversity_score']}")
        print(f"Content-Based Diversity Score: {result['content_based_diversity_score']}")
        print(f"KMeans Novelty Score: {result['kmean_novelty_score']}")
        print(f"Content-Based Novelty Score: {result['content_based_novelty_score']}")
        print(f"KMeans Accuracy: {result['kmean_accuracy']}")
        print(f"Content-Based Accuracy: {result['content_based_accuracy']}")


if __name__ == '__main__':
    main()
