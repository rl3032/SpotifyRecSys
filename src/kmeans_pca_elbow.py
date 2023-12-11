"""
K-means clustering algorithm

Author: RJ
Date: 12/10/2023

"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def perform_pca(data_scaled, n_components=2):
    """
    Perform PCA on standardized data.

    Args:
        data_scaled: Numpy array of standardized features.
        n_components: Number of components for PCA.
    
    Returns:
        Numpy array of PCA data.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data_scaled)


def plot_pca_kmeans(pca_data, labels):
    """
    Plot PCA data with KMeans labels.

    Args:
        pca_data: Numpy array of PCA data.
        labels: Numpy array of KMeans labels.
    
    Returns:
        None.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pca_data[:, 0], y=pca_data[:, 1], mode='markers', marker=dict(color=labels)))
    fig.update_layout(title='KMeans Clustering', xaxis_title='PCA1', yaxis_title='PCA2')
    fig.show()


def perform_elbow_method(data_scaled, range_clusters):
    """
    Perform elbow method on standardized data.

    Args:
        data_scaled: Numpy array of standardized features.
        range_clusters: Range of clusters to use.
    
    Returns:
        List of inertia values.
    """
    inertia = []
    for i in range(1, range_clusters+1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)
    return inertia


def plot_elbow(inertia):
    """
    Plot elbow method.

    Args:
        inertia: List of inertia values.
    
    Returns:
        None.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, 11)), y=inertia, mode='lines+markers'))
    fig.update_layout(title='Elbow Method', xaxis_title='Number of Clusters', yaxis_title='Inertia')
    fig.show()


def main():
    # Load the dataset
    data = pd.read_csv('data/processed/final_data.csv')

    # Define features for clustering
    features = ['popularity', 'duration_ms', 'explicit', 'acousticness', 'danceability', 'energy', 
                'instrumentalness', 'liveness', 'speechiness', 'valence', 'key', 'loudness', 'mode', 'tempo', 'time_signature']

    # Standardize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    pca_data = perform_pca(data_scaled)
    labels = KMeans(n_clusters=4).fit_predict(data_scaled)

    plot_pca_kmeans(pca_data, labels)
    inertia = perform_elbow_method(data_scaled, range_clusters=10)
    plot_elbow(inertia)

if __name__ == '__main__':
    main()
