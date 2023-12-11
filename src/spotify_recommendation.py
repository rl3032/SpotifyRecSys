import requests
import base64
import configparser
import pandas as pd
import csv

# Load Spotify API credentials from config file
config = configparser.ConfigParser()
config.read('../config.ini')
client_id = config['SPOTIFY']['CLIENT_ID']
client_secret = config['SPOTIFY']['CLIENT_SECRET']


# Get access token
def get_access_token(client_id, client_secret):
    try:
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())

        token_url = "https://accounts.spotify.com/api/token"
        token_data = {
            "grant_type": "client_credentials"
        }
        token_headers = {
            "Authorization": f"Basic {client_creds_b64.decode()}"
        }

        r = requests.post(token_url, data=token_data, headers=token_headers)
        token_response_data = r.json()
        return token_response_data["access_token"]
    except requests.RequestException as e:
        print(f"Error getting access token: {e}")
        raise


# Search Spotify ID
def get_track_id(track_name, access_token, artist_name=None, release_year=None):
    try:
        search_query = f"track:{track_name}"
        if artist_name:
            search_query += f" artist:{artist_name}"

        search_url = "https://api.spotify.com/v1/search"
        search_params = {
            "q": search_query,
            "type": "track",
            "limit": 50
        }
        search_headers = {
            "Authorization": f"Bearer {access_token}"
        }

        search_response = requests.get(search_url, headers=search_headers, params=search_params)
        if search_response.status_code != 200:
            raise Exception(f"Search API returned error: {search_response.status_code} - {search_response.text}")

        search_data = search_response.json()
        tracks = search_data['tracks']['items']

        # filter tracks to match the year condition if release_year is provided
        if release_year:
            tracks = [track for track in tracks if track['album']['release_date'].startswith(str(release_year))]

        if not tracks:
            raise Exception(
                f"No track found for the given name '{track_name}' with specified criteria. Please check the details or try a different combination.")

        return tracks[0]['id']
    except Exception as e:
        print(f"Error searching track ID: {e}")


# Function to get recommendations from Spotify API
def get_recommendations(track_id, access_token, total_songs=500, market=None):
    recommendations = []
    url = "https://api.spotify.com/v1/recommendations"

    while len(recommendations) < total_songs:
        params = {
            "limit": min(100, total_songs - len(recommendations)),  # Request maximum of 100 songs at a time
            "seed_tracks": track_id
        }
        if market:
            params["market"] = market

        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        recommendations.extend(data['tracks'])

    return recommendations


# Function to read CSV and return a DataFrame
def read_final_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def filter_songs(recommendations, final_data_df):
    # final_data_set = set(zip(final_data_df['name'], final_data_df['year'].astype(str)))
    final_data_set = set(zip(final_data_df['name'], final_data_df['year'].astype(str), final_data_df['artists']))
    filtered_songs = []
    for song in recommendations:
        song_name = song['name']
        release_year = song['album']['release_date'][:4]
        artists = ', '.join(artist['name'] for artist in song['artists'])

        if (song_name, release_year, artists) in final_data_set:
            filtered_songs.append(song)

    return filtered_songs


# Function to write the filtered songs to a new CSV file
def write_filtered_songs_to_csv(songs, file_path: str):
    seen_songs = set()
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['song_name', 'year', 'artists'])
        writer.writeheader()
        for song in songs:
            song_tuple = (song['name'], song['album']['release_date'][:4], ', '.join(artist['name'] for artist in song['artists']))
            if song_tuple not in seen_songs:
                writer.writerow({'song_name': song['name'], 'year': song['album']['release_date'][:4],
                                 'artists': ', '.join(artist['name'] for artist in song['artists'])})
                seen_songs.add(song_tuple)


def main():
    try:
        access_token = get_access_token(client_id, client_secret)
        song_name = "Shape of You"
        # artist_name = "One Direction"
        # release_year = 2013
        # track_id = get_track_id(song_name, access_token, artist_name, release_year)
        track_id = get_track_id(song_name, access_token)
        recommendations = get_recommendations(track_id, access_token, 500)
        final_data_df = read_final_data('../data/processed/final_data.csv')
        filtered_songs = filter_songs(recommendations, final_data_df)
        for song in filtered_songs:
            song_name = song['name']
            release_year = song['album']['release_date'][:4]
            artists = ', '.join(artist['name'] for artist in song['artists'])
            print(f"Song Name: {song_name}, Year: {release_year}, Artists: {artists}")
        write_filtered_songs_to_csv(filtered_songs, '../data/processed/filtered_songs.csv')
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
