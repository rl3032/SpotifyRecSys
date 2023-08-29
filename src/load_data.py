import pandas as pd


def load_data(filepath):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    df (pd.DataFrame): DataFrame containing the loaded data.
    """
    df = pd.read_csv(filepath)
    return df

def get_song_index(name, data):
    """
    Retrieves the index of a song in the dataset based on its name.

    Parameters:
    name (str): Name of the song to search for.
    data (pd.DataFrame): DataFrame containing the song data.

    Returns:
    song_index (int or None): Index of the song if found, or None if not found.
    """
    indices = data[data['name'] == name].index

    if len(indices) > 0:
        return indices[0]
    else:
        return None
