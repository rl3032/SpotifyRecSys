from sklearn.neighbors import NearestNeighbors

def train_knn_model(X, metric='cosine', algorithm='brute'):
    """
    Trains a k-Nearest Neighbors (k-NN) model on the given data.

    Parameters:
    X (array-like or sparse matrix): The data to train the k-NN model on.
    metric (str, optional): The distance metric to use for k-NN. Default is 'cosine'.
    algorithm (str, optional): The algorithm to use for k-NN. Default is 'brute'.

    Returns:
    model (NearestNeighbors): Trained k-NN model.
    """
    model = NearestNeighbors(metric=metric, algorithm=algorithm)
    model.fit(X)
    return model


def recommend_songs(model, song_index, X, n_neighbors=10):
    """
    Recommends similar songs using a trained k-NN model.

    Parameters:
    model (NearestNeighbors): Trained k-NN model.
    song_index (int): Index of the song for which recommendations are to be made.
    X (array-like or sparse matrix): The data used to train the k-NN model.
    n_neighbors (int, optional): Number of neighbors to consider for recommendations. Default is 10.

    Returns:
    recommend_song_indices (array): Array of indices of recommended songs.
    """
    query_point = X.iloc[song_index, :].values.reshape(1, -1)
    distances, indices = model.kneighbors(query_point, n_neighbors=n_neighbors+1) 
    recommend_song_indices = indices.flatten()[1:]
    return recommend_song_indices