# Spotify Song-Based Recommendation

## Table of Contents
 1. [Introduction](#introduction)
 2. [Data Understanding](#data-understanding)
 3. [Exploratory Data Analysis](#exploratory-data-analysis)
 4. [Feature Preprocessing](#feature-preprocessing)
 5. [Model Development](#model-development)
 6. [Model Evaluation](#model-evaluation)
 7. [Result and Discussion](#results-and-discussion)
 8. [Conclusion and Future Work](#conclusion-and-future-work)
 9. [References](#references)
 10. [Appendix](#10-appendix-tables-and-figures)

## 1. Introduction
In the era of digital music consumption, personalized recommendation systems have become increasingly important to user experience. Leveraging the vast and diverse Spotify dataset, our project aims to develop a song-based recommendation system.

## 2. Data Understanding
In 2020, Yamaç Eren Ay released an extensive Spotify dataset on Kaggle, encompassing over 160,000 songs from the years 1921 to 2020 [1]. In the following year, Yamaç Eren Ay released a larger track dataset that contains over 500,000 songs [1]. Motivated by this valuable resource, our team decided to utilize this new dataset, hereafter referred to as `data`, for developing our music recommendation system. Our initial step involved a comprehensive analysis of the song attributes in `data`, along with a detailed understanding of feature definitions obtained from the Spotify Web API.

We used the `data.shape` method to confirm that data contains 586,672 tracks across 20 features. Additionally, by employing the `data.info()` method, we discovered that within `data`, 9 features are of float type, 6 are integers, and 5 are strings. To provide further clarity, we present a detailed feature definition table, which draws information from the dataset and the Spotify API documentation [2][3]:


| Feature Name     | Type    | Description |
| ------------     | ------- | ----------- |
| name             | string  | The name of the track. |
| artists          | string  | The artists(s) of the track. |
| release_date     | string  | The specific date when the track was released.
| popularity       | integer | The popularity of the track. The value will be between 0 and 100, with 100 being the most popular. |
| duration_ms      | integer | Length of the track in milliseconds. |
| explicit         | integer | Indicates if the track has explicit lyrics (1 = yes, 0 = no or unknown). |
| id               | string  | The Spotify ID for the track. |
| id_artists       | string  | The Spotify ID for the artist. |
| acousticness     | float   | A measure from 0.0 to 1.0 indicating how acoustic the track is. |
| danceability     | float   | Ranges from 0.0 to 1.0, indicating how suitable a track is for dancing. |
| energy           | float   | A measure from 0.0 to 1.0 indicating the intensity and activity level of a track. |
| instrumentalness | float   | Predicts if a track is instrumental, with values closer to 1.0 indicating no vocals. |
| key              | integer | The musical key of the track, with integers representing Pitch Class notation. -1 if undetected. |
| liveness         | float   | Measures the likelihood of the track being performed live, with higher values indicating higher probability. |
| loudness         | float   | The average loudness of the track in decibels (dB). |
| mode             | integer | Indicates the modality of the track, with 1 for major and 0 for minor. |
| speechiness      | float   | Measures the presence of spoken words, with higher values indicating more speech. |
| tempo            | float   | The tempo of the track in beats per minute (BPM). |
| valence          | float   | Measures the musical positiveness of a track, ranging from 0.0 to 1.0. |
| time_signature   | integer   | Refers to the number of beats in a measure, or bar, of the music.

Based on the feature definitions provided in the Spotify Documentation, we have categorized them into two distinct groups: `track_features` and `audio_features`. The `track_features` group encompasses all the attributes directly associated with the track itself, such as its name, artists, and Spotify ID. In our classification, we have identified 8 features that fall under `track_features`. The figure below presents the `track_features` for the last five tracks in our dataset.

<figure>
    <img src="/report/image/track_feature_table.jpg" 
    alt="Track Feature Table">
    <figcaption style="text-align: center;">Figure 1: Track Features for Lastest 5 tracks</figcaption>
</figure>


Meanwhile, the remaining 12 features are categorized as    `audio_features`. These pertain to the musical and acoustic properties of the tracks. The subsequent figure illustrates the `audio_features` for the last five tracks in our dataset.


<figure>
    <img src="/report/image/audio_feature_table.jpg" 
    alt="Audio Feature Table">
    <figcaption style="text-align: center;">Figure 2: Audio Features for Lastest 5 tracks</figcaption>
</figure>


From above two figures, we noticed that some features are not normalized, such as `key` and `loudness`. Therefore, we will further categorized our feature groups into `track_features`, `audio_features_normalized`, and `audio_features_not_normalized`.

```
track_features = ['name', 'artists', 'year', 'release_date','popularity', 'duration_ms', 'explicit', 'id']
audio_features_normalized = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
audio_features_not_normalized = ['key', 'loudness', 'mode', 'tempo', 'time_signature']
```

Prior to moving to the data analysis phase, it is crucial to ascertain the completeness of the `data`. Upon examination, it is observed that 71 tracks in the dataset contain missing values in the `name` column. Given that these constitute a minor fraction of the overall dataset, we opt to employ the `data.dropna()` method. This approach effectively removes these incomplete entries, thereby ensuring the integrity and cleanliness of the data for subsequent analysis.

## 3. Exploratory Data Analysis
Exploratory Data Analysis (EDA) plays a pivotal role in comprehending the intricacies of dataset features, discerning underlying patterns and anomalies, and setting the stage for effective model development. In our analysis, we will delve into various dimensions, including temporal trends (time-based analysis), categorical distinctions (genres), and individual contributors (artists). This multifaceted approach ensures a comprehensive understanding of the dataset, paving the way for more informed and accurate modeling.

### 3.1 Music Over Time
Music trends evolve over time. By analyzing data over different periods, we can track changes in popularity, the rise and fall of certain musical styles, or the emergence of new ones. For music streaming services like Spotify, understanding historical trends is vital for improving recommendation algorithms. Analyzing how user's preferences changes over time helps in creating more personalzied and dynamic playlists.

We organize the `data` by categorizing it into annual segments, and then calculate the mean for each year. This process results in the creation of `year_data`, a dataset that encapsulates the average values for each year.

<figure>
    <img src="/report/image/popularity_over_year.jpg"
    alt="Popularity Trend Analysis">
    <figcaption style="text-align:center">Figure 3: Track Popularity Over Time
    </figcaption>
</figure>

To start with, we analyzed the popularity of tracks over time, from 1922 to the 2020, using the `year_data`. From Figure 3, we have several observations:
- low popularity in eary years
- steady growth post-1950s
- acceleration in growth from the 1960s
- high popularity in modern times
- overall upward trend

<figure>
    <img src="/report/image/audio_feature_over_year.jpg"
    alt="Audio Features Changes Over Time">
    <figcaption style="text-align:center">Figure 4: Audio Features Changes Over Time
    </figcaption>
</figure>

Figure 4 plots several audio features of music tracks over time. Each line represents a different audio features as defined by Spotify, and the y-axis indicates the value of the feature, whice is normalized between 0 and 1. And we have following observations:
- **Acousticness**: The feature for acousticness starts very high in the 1920s and sees a steep decline around 1950s. This likely correlates with the advent of electric and electronic instruments in music production.
- **Danceability**: There is a slight upward trend from the 1960s onward, suggesting that tracks have become more dance-friendly over time.
- **Energy**: The energy level has an upward trend starting from 1950s, which could reflect the rise of more vibrant and dynamic generes like rock and roll and later, electronic and pop music.
- **Instrumentalness**:  Instrumentalness spikes in certain periods, such as the 1930s and around the 1970s.
- **Liveness**: The liveness feature has a general downward trend.
- **Speechiness**: Speechiness remains relatively low throughout, with some variation.
- **Valence**: Valence, which measures the positivity conveyed by a track, starts off high in the 1920s but generally trends downward over time. The overall trend might suggest that songs have become less 'happy' sounding.

For a comprehensive examination of additional trend patterns across various audio features, please refer to Appendix A: Extended Trend Analysis.

### 3.2 Music By Genres
Different genres appeal to different audiences. Streaming services like Spotify rely on sophisticated algorithms to recommend music to users. Understanding the nuances of genre-specific tracks helps to improve user experience by delivering more accurate recommendations.

<figure>
    <img src="/report/image/genres_by_popularity.jpg"
    alt="Top 15 Genres By Popularity">
    <figcaption style="text-align:center">Figure 5: Top 15 Genres By Popularity
    </figcaption>
</figure>

This figure includes top 15 genres from a diverse set. The genres reflect a range of cultural influences, from "south african house" to "turkish edm" and "chinese electropop." 

Some of the genres, like "basshall" or "trap venezolano," might represent emerging styles that have gained a following. The presence of genres like "pagode baiano" and "ritmo kombina" indicates regional music styles, which might be extremely popular locally but less known on a global scale.


<figure>
    <img src="/report/image/valence_distribution_among_genres.jpg"
    alt="Valence Distribution for Top 15 Genres">
    <figcaption style="text-align:center">Figure 6: Valence Distribution for Top 15 Genres
    </figcaption>
</figure>

The bar graph in Figure 6 show the distribution of the valence audio feature across different music genres. Valence, in Spotify's audio feature terminology, measures the musical positiveness conveyed by a track. Higher valence indicates a more positive, cheerful, and euphoric sound, while lower valence would suggest a more negative, sad, or angry vibe. Here are some observations based on the graph:
- **Variation Across Genre**: There is a clear variation in valence across the genre shown, which suggests that different genres convey different emotional tones.
- **High Valence Genres**: Some genres, such as "south african house" and "afroswing," have particularly high valence, suggesting that these genres typically feature music that is more upbeat and positive in nature.
- **Low Valence Genres**: Conversely, genres like "chinese electropop" and "indie triste" have lower valence scores, indicating these genres may often have more somber or subdued emotional content.
- **Cultural Emotional Expressions**: The emotional tone conveyed by these genres could reflect cultural expressions of emotion through music. For example, "trap venezolano" and "pagode baiano" might have distinct emotional signatures that resonate with the cultural contexts they originate from.

For an in-depth analysis of the distribution patterns among the top 15 genres, please consult Appendix B: Extended Genre Distribution Analysis.

### 3.3 Music By Artists
Data on artists and their tracks feed recommendation engines, helping platforms like Spotify suggest new music to users based on their listening habits, thus improving the user experience.

<figure>
    <img src="/report/image/top_15_artists_of_2017.jpg"
    alt="Top 15 Artists by Popularity in 2017">
    <figcaption style="text-align:center">Figure 7: Top 15 Artists by Popularity in 2017
    </figcaption>
</figure>

<figure>
    <img src="/report/image/top_15_songs_of_2017.jpg"
    alt="Top 15 Songs by Popularity in 2017">
    <figcaption style="text-align:center">Figure 8: Top 15 Songs by Popularity in 2017
    </figcaption>
</figure>
Figure 7 presents a bar graph that vividly showcases the popularity of the Top 15 artists in 2017. In this graph, each artist is represented by a bar, with the bar's length directly correlating to their respective popularity scores. Similarly, Figure 8 features a bar graph depicting the popularity of the Top 15 songs from the same year, where each song is represented by a bar whose length signifies its popularity score.

Based on these graphs, it can be inferred that the artists and songs with the highest popularity scores are likely the ones that garnered the most attention and listenership in 2017.

<figure>
    <img src="/report/image/top_artists_in_last_15_years.jpg"
    alt="Top Artists in Last 15 years">
    <figcaption style="text-align:center">Figure 8: Top Artists by Decades
    </figcaption>
</figure>

Figure 9 displays a comprehensive table showcasing the top artists for each year from 2007 to 2021, along with their corresponding popularity ratings. Notably, there's a discernible upward trend in these popularity scores, culminating in 2021, which features artists reaching the peak score of 100.0.

Comparing this with the insights from the previous three figures, we recognize familiar names such as Lady Gaga, Billie Eilish, and Ed Sheeran. Additionally, we identify hit songs like 'Believer', 'Perfect', and 'Something Just Like This'. This overlap of artists and songs across different years and metrics leads to the conclusion that these artists and songs have not only achieved high popularity scores but have also maintained a significant presence and influence in the music industry over this period. Also, the artist analysis implies that our dataset is comprehensive and ready for building machine learning algorithms.

## 4. Feature Preprocessing


## 5. Model Development


## 6. Model Evaluation

## 7. Result and Discussion

## 8. Conclusion and Future Work

## 9. References
[1] Y. E. Ay, "Spotify Dataset 1921-2020, 600k+ Tracks," Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks. Accessed on: November 21, 2023.
[2] Spotify. "Get Track." Spotify for Developers. [Online]. Available: https://developer.spotify.com/documentation/web-api/reference/get-track. Accessed on: November 22, 2023.
[3] Spotify. "Get Several Audio Features." Spotify for Developers. [Online]. Available: https://developer.spotify.com/documentation/web-api/reference/get-several-audio-features. Accessed on: November 22, 2023.

## 10. Appendix: Tables and Figures

### Appendix A: Extended Trend Analysis

<figure>
    <img src="/report/image/loudness_changes_over_year.jpg"
    alt="Loudness Changes Over Time">
    <figcaption style="text-align:center">Figure A1: Loudness Changes Over Time
    </figcaption>
</figure>


<figure>
    <img src="/report/image/tempo_changes_over_year.jpg"
    alt="Tempo Changes Over Time">
    <figcaption style="text-align:center">Figure A2: Tempo Changes Over Time
    </figcaption>
</figure>


<figure>
    <img src="/report/image/key_changes_over_year.jpg"
    alt="Key Changes Over Time">
    <figcaption style="text-align:center">Figure A3: Key Changes Over Time
    </figcaption>
</figure>

### Appendix B: Extended Genre Analysis

<figure>
    <img src="/report/image/danceability_distribution_among_genres.jpg"
    alt="Danceability Distribution for Top 15 Popular Genres">
    <figcaption style="text-align:center">Figure B1: Danceability Distribution for Top 15 Popular Genres
    </figcaption>
</figure>


<figure>
    <img src="/report/image/energy_distribution_among_genres.jpg"
    alt="Engery Distribution for Top 15 Popular Genres">
    <figcaption style="text-align:center">Figure B2: Energy Distribution for Top 15 Popular Genres
    </figcaption>
</figure>


<figure>
    <img src="/report/image/acousticness_distribution_among_genres.jpg"
    alt="Acousticness Distribution for Top 15 Popular Genres">
    <figcaption style="text-align:center">Figure B3: Acousticness Distribution for Top 15 Popular Genres
    </figcaption>
</figure>


<figure>
    <img src="/report/image/instrumentalness_distribution_among_genres.jpg"
    alt="Instrumentalness Distribution for Top 15 Popular Genres">
    <figcaption style="text-align:center">Figure B4: Instrumentalness Distribution for Top 15 Popular Genres
    </figcaption>
</figure>


