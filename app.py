import streamlit as st
from src.load_data import load_data, get_song_index
from src.model import train_knn_model, recommend_songs

# set the looking font
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.title("Spotify Recommendation System")
    st.markdown("Welcome to your personalized music recommender engine!\
                You can input the name of a song and get a recommendation list.")
    
    # Load dataset
    data = load_data("data/data.csv")

    song_features_normalized = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
    song_features_not_normalized = ['duration_ms', 'key', 'loudness', 'mode', 'tempo']

    all_features = song_features_normalized + song_features_not_normalized + ['year', 'popularity']

    X = data[all_features]

    model = train_knn_model(X)

    # Add selections to the sidebar
    st.sidebar.markdown("### Number of Recommendations")
    num_rec_songs = st.sidebar.slider("Select the number of songs in your list", 5, 30, 10)
    

    # Provide a text area for the user to enter a seed song
    song_name = st.text_area("Enter a song name")
    
    if st.button("Recommend"):
        song_index = get_song_index(song_name, data)
        recommend_song_indices = recommend_songs(model, song_index, X, num_rec_songs)
        recommend_song_list = data[['name', 'artists', 'year', 'popularity']].iloc[recommend_song_indices]
        st.write(recommend_song_list)  
    

if __name__ == "__main__":
    main()