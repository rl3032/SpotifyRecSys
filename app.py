import streamlit as st
import numpy as np
import pandas as pd


# Load the dataset
data = pd.read_csv("data/data.csv")

song_features_normalized = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
song_features_not_normalized = ['duration_ms', 'key', 'loudness', 'mode', 'tempo']

all_features = song_features_normalized + song_features_not_normalized + ['decade', 'popularity']

# set a good looking font
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
    
    # Add selections to the sidebar
    st.sidebar.markdown("### Number of Recommendations")
    num_rec_songs = st.sidebar.slider("Select the number of songs in your list", 
                                      5, 30, 10)
    

    # Provide a text area for the user to enter a seed song
    st.markdown("## Try to generate a recommendation list")
    song_name = st.text_area("Enter the name of a song")

if __name__ == "__main__":
    main()