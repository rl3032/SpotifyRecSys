import requests
import base64
import configparser

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


# Get spotify recommendations
def get_recommendations(track_id, access_token, limit=30, market=None):
    try:
        recommendations_url = "https://api.spotify.com/v1/recommendations"
        recommendations_params = {
            "limit": limit,
            "seed_tracks": track_id
        }
        if market:
            recommendations_params["market"] = market

        recommendations_headers = {
            "Authorization": f"Bearer {access_token}"
        }

        recommendations_response = requests.get(recommendations_url, headers=recommendations_headers, params=recommendations_params)
        return recommendations_response.json()
    except Exception as e:
        print(f"Error getting recommendations: {e}")


# Handle result
def extract_track_info(recommendations_json):
    track_info_list = []

    for track in recommendations_json['tracks']:
        track_name = track['name']
        artists = ', '.join(artist['name'] for artist in track['artists'])
        release_date = track['album']['release_date']

        track_info = {
            'song_name': track_name,
            'artists': artists,
            'release_date': release_date
        }

        track_info_list.append(track_info)

    return track_info_list


def main():
    try:
        access_token = get_access_token(client_id, client_secret)
        song_name = "Story of My Life"
        artist_name = "One Direction"
        release_year = 2013
        # track_id = get_track_id(song_name, access_token, artist_name, release_year)
        track_id = get_track_id(song_name, access_token)
        recommendations = get_recommendations(track_id, access_token)
        track_info_list = extract_track_info(recommendations)
        for info in track_info_list:
            print(f"Song Name: {info['song_name']}, Artists: {info['artists']}, Release Date: {info['release_date']}")
        return track_info_list
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
