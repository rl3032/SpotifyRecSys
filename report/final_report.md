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
    <figcaption style="text-align:center">Figure 9: Top Artists from 2007 to 2021
    </figcaption>
</figure>

Figure 9 displays a comprehensive table showcasing the top artists for each year from 2007 to 2021, along with their corresponding popularity ratings. Notably, there's a discernible upward trend in these popularity scores, culminating in 2021, which features artists reaching the peak score of 100.0.

Comparing this with the insights from the previous three figures, we recognize familiar names such as Lady Gaga, Billie Eilish, and Ed Sheeran. Additionally, we identify hit songs like 'Believer', 'Perfect', and 'Something Just Like This'. This overlap of artists and songs across different years and metrics leads to the conclusion that these artists and songs have not only achieved high popularity scores but have also maintained a significant presence and influence in the music industry over this period. Also, the artist analysis implies that our dataset is comprehensive and ready for building machine learning algorithms.

## 4. Feature Preprocessing

Following our initial data understanding and exploratory data analysis (EDA), the next step is to preprocess the data to ensure it's optimally prepared for model development.

During our review, we identified several areas for preprocessing. Firstly, we observed that certain audio features have yet to be normalized. Normalizing these features is crucial for maintaining consistency across the dataset. Secondly, the `release_date` field offers an opportunity for simplification; we plan to convert this to just the year. Lastly, we noted that the `artists` is currently stored as a list of artist names. To streamline our analysis, we'll need to process this data into a more manageable format.

<figure>
    <img src="/report/image/final_feature_stat.jpg"
    alt="Final Feature Statistics">
    <figcaption style="text-align:center">Figure 10: Final Feature Statistics
    </figcaption>
</figure>

In Figure 11, we calcualte the pairwise correlation efficients between all features. Here is the analysis of the matrix:

- `loudness` and `energy` have a high positive correlation of 0.76, suggesting that louder tracks tend to be perceived as more energetic. 
- `acousticness` has a strong negative correlation with `energy` (-0.71) and `loudness` (-0.52), indicating that acoustic tracks are generally quieter and less energetic.
- `danceability` and `valence` are positively correlated (0.53), which can imply that songs perceived as more danceable are often happier or more positive.

<figure>
    <img src="/report/image/feature_correlation_matrix.jpg"
    alt="Final Feature Statistics">
    <figcaption style="text-align:center">Figure 11: Feature Correlation Matrix
    </figcaption>
</figure>

Although it is standard practice to exclude features with high correlation to prevent multicollinearity in our models, we have chosen to retain the current set of features. This decision is strategic, allowing us the flexibility to directly align our model with the Spotify API for future testing and validation purposes.

## 5. Model Development

Our data is centered around an unsupervised machine learning challenge. To address this, our primary approach involves deploying the k-means clustering algorithm. We will determine the ideal number of clusters (K) using the elbow method and employ dimensionality reduction techniques like t-SNE for effective visualization of the clusters. Additionally, our secondary model will implement a content-based filtering approach, a classic recommendation algorithm, using cosine similarity to identify songs closely resembling our chosen seed song.

### 5.1 Euclidean and Cosine Distance
Both K-Means Clustering and Content-Based Filtering algorithms fundamentally rely on distance measures to assess similarity or dissimilarity between data points. Among the common distance measures, Euclidean and Cosine distances are notably prevalent.

The **Euclidean distance**, often referred to as the straight-line distance, is calculated between two points in a Euclidean space. This measure is derived from the Cartesian coordinates of the points and is particularly intuitive and straightforward to implement. In an n-dimensional Euclidean space, the Euclidean distance $d$ between two points $p$ and $q$, each having n coordinates, is given by the formula:

\[ d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2} \]

This distance measure is simple yet effective, producing excellent results in various applications, especially when applied to low-dimensional data. Algorithms such as K-Means often exhibit superior performance when Euclidean distance is utilized in such contexts [4].

The **Cosine distance**, commonly referred to as the Cosine similarity, is a measure used to determine the similarity between two non-zero vectors. This metric calculates the cosine of the angle between two vectors in a multi-dimensional space, providing a measure of their orientation, regardless of their magnitude [5].

Mathematically, the cosine similarity $cos(θ)$ between two n-dimensional vectors, $A$ and $B$, is expressed through the dot product and magnitude. This relationship is represented as follows, where $A_{i}$ and $B_{i}$ are the i-th components of vectors $A$ and $B$, respectively [5]:

\[ \cos(\theta) = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}} \]

The value of cosine similarity ranges between -1 and 1. A value of -1 indicates that the two vectors are diametrically opposite, 0 signifies no correlation, and 1 indicates that the vectors are identical [6]. Values between these extremes denote varying degrees of similarity or dissimilarity.

<figure>
    <img src="/report/image/different_angles_of_cosine_similarities.jpg"
    alt="Different Angles of Cosine Similarities">
    <figcaption style="text-align:center">Figure 12: Different Angles of Cosine Similarities [6]
    </figcaption>
</figure>

In clustering contexts, cosine similarity is particularly useful when the orientation or direction of the vector is more relevant than its magnitude.

The main difference between Euclidean distance and Cosine similarity is their sensitivity towards the magnitude of vectors: Cosine similarity completely ignores the magnitude of vectors and only cares about the angle between them, while Euclidean distance cares about both the magnitude and direction of the vectors. 

<figure>
    <img src="/report/image/euclidean_and_cosine_distances.jpg"
    alt="Euclidean and Cosine Distances">
    <figcaption style="text-align:center">Figure 13: Euclidean and Cosine Distances [4]
    </figcaption>
</figure>

A mathematical example to illustrate would be: If there is a vector with a larger magnitude in the same direction as a vector with a shorter magnitude, they would have a cosine similarity of 1, indicating two vectors are the same (in direction), even though the Euclidean distance between them would be quite large, reflecting the substantial difference in their magnitudes [7].

The difference in the sensitivity towards the magnitude of vectors results in the difference of their applications: Cosine similarity is widely used in cases like text analysis where the frequency of occurrence of terms can often be more relevant than their absolute counts. On the other hand, Euclidean distance can be more appropriate when data is dense or when the magnitude of vectors is significant [8].

### 5.2 K-Mean Clustering
**Clustering**, a fundamental unsupervised machine learning technique, involves segmenting a dataset into distinct groups or clusters [9]. In this process, data points within each cluster demonstrate similar characteristics or behaviors, making them more akin to each other than to points in different clusters. Clustering algorithms primarily group data based on specific similarities [10].

**K-means** is an essential and efficient clustering algorithm in unsupervised machine learning, used for dividing datasets into K distinct, non-overlapping subgroups or clusters. 

<figure>
    <img src="/report/image/k-means_clustering_illustration.jpg"
    alt="K-Means Clustering Illustration">
    <figcaption style="text-align:center">Figure 14: K-Means Clustering Illustration [9]
    </figcaption>
</figure>

The process initiates by selecting K points as cluster centroids randomly. Each data point is then assigned to its nearest centroid, based on Euclidean distance, and centroids are recalculated by averaging the points in each cluster [11]. This assignment and update process iterates until the centroids stabilize and the data points' assignments to clusters cease changing. 


#### Steps:

1. **Specify the number `k` of clusters to assign.**
   - `k` is a hyperparameter that defines the number of clusters to be created.
   
2. **Randomly initialize `k` centroids.**
   - The centroids are the initial guesses for the location of the cluster centers.
   
3. **Repeat the following steps until the centroid positions do not change:**

   a. **Expectation:** 
      - Assign each point to its closest centroid.
      - This creates a "cluster assignment" for each point.

   b. **Maximization:** 
      - Compute the new centroid (mean) of each cluster.
      - The mean is calculated by averaging the points in each cluster.

4. **Termination Condition:**
    - The algorithm stops iterating when the centroids do not move significantly, indicating that the clusters are stable.

K-means assumes spherical clusters of similar size and can be sensitive to the initial centroid positions, often requiring multiple runs for robustness. Choosing the correct number of clusters, K, is crucial, with methods like the Elbow method and Silhouette analysis assisting in this determination [9]. However, the algorithm's effectiveness can be compromised by outliers, as they can significantly influence centroid positioning.

### 5.3 Elbow Method
The **elbow method** is a visual technique used in determining the optimal number of clusters, *K*, for K-means clustering. This method involves calculating the **Within-Cluster Sum of Squares (WCSS)** for various cluster counts. WCSS is the total squared distance between each point in a cluster and the cluster's centroid [12]. We employ the following Python code, utilizing the `sklearn`.cluster module, to implement the elbow method:

```python
from sklearn.cluster import KMeans

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
```

By varying K from 1 to `range_clusters` and plotting WCSS against each *K* value, a distinctive "elbow" shape often emerges in the graph. This shape occurs because the WCSS tends to decrease as *K* increases, with the most significant drop typically happening at the optimal *K*. Initially, when *K* equals 1, WCSS is at its maximum. As *K* increases, there's a sharp decrease in WCSS, creating a bend in the graph resembling an elbow. Beyond this point, the graph tends to level off, indicating diminishing returns in reducing WCSS with further increases in *K* [12]. The *K* value at this "elbow point" is generally considered the most suitable choice for the number of clusters. For our data, the optimal *K* value is 4.

<figure>
    <img src="/report/image/elbow_method.jpg"
    alt="Perform Elbow Method to Choose Optimal K">
    <figcaption style="text-align:center">Figure 15: Perform Elbow Method to Choose Optimal K
    </figcaption>
</figure>

### 5.4 Principal Component Analysis
Principal Component Analysis (PCA) is a cornerstone technique in dimensionality reduction, extensively utilized in data analysis and machine learning [13]. The essence of PCA lies in its ability to discern and highlight patterns in the relationships between variables, subsequently representing these patterns with fewer, more potent variables known as principal components. The PCA procedure unfolds through several methodical steps:

1. **Standardization:** Given that input variables often vary in scale, standardizing these variables is crucial. This process involves adjusting the data to have a mean of zero and a unit variance, ensuring each variable contributes equitably to the analysis.

2. **Covariance Matrix Calculation:** Post-standardization, the next stride is computing the covariance matrix for the standardized data. This matrix captures the pairwise relationships between variables, offering insights into their joint variability.

3. **Eigenvalue Decomposition:** The covariance matrix undergoes eigenvalue decomposition to extract its eigenvectors and eigenvalues. The eigenvectors delineate the principal components, essentially the new axes of data variation, while the eigenvalues quantify the variance captured by each principal component. These eigenvectors are ranked in descending order of their corresponding eigenvalues.

4. **Selection of Principal Components:** Choosing the right number of principal components is a strategic decision. It's typically guided by the cumulative explained variance criterion, where principal components are selected based on their collective contribution to the total variance (e.g., 95% or 99%).

5. **Projection:** The final step involves projecting the original data onto the new feature subspace formed by the selected principal components. This transformation, achieved by multiplying the standardized data with the matrix of chosen eigenvectors, results in a dataset with reduced dimensionality.

PCA is particularly useful in K-means clustering.  By reducing the number of dimensions without significant loss of information, PCA simplifies the clustering process, making it computationally more efficient and less prone to overfitting [14].

<figure>
    <img src="/report/image/pca_result.jpg"
    alt="Perform PCA Dimension Reduction">
    <figcaption style="text-align:center">Figure 16: Perform PCA Dimension Reduction
    </figcaption>
</figure>

In our analysis, we applied the outlined steps to our training dataset, which consists of 15 features. We successfully reduced the data to a two-dimensional space, grouping it into four distinct clusters. This optimal number of clusters, `K = 4`, was determined through the application of the elbow method.

### 5.5 Content-based Filtering
Generally, recommender systems are algorithms designed to suggest the most suitable items (which could be movies to watch, text to read, products to buy, etc.) to users, tailored to their unique preferences. Such systems could simplify the user's decision-making process by narrowing down choices from a vast number of options. There are two fundamental paradigms used in recommender systems: Collaborative Filtering and Content-Based Filtering [15].

1. **Collaborative Filtering** is based on the principle that users with similar past preferences will probably have similar tastes in the future, so the system generates recommendations using only the past interactions recorded between users and items.

2. **Content-Based Filtering** methods recommend items to users relying on both the users’ preferences and the features of the items they have interacted with.

Since our dataset only contains song tracks related information, we will use the **Item-Centered Approach** from the Content-Based Filtering. This method focuses on leveraging the intrinsic characteristics of songs, such as tempo, rhythm, and lyrics. This method, while efficiently utilizing song features and metadata to find patterns and similarities between tracks, does face limitations. 

Primarily, it lacks personalization, as recommendations are based on general song attributes rather than individual user preferences and interactions. This might lead to over-specialization, where users are continually presented with songs too similar to their past choices, potentially causing a monotonous user experience. Additionally, new songs with limited data might be overlooked, and the system might struggle with introducing variety and serendipity into the recommendations.

### 5.6 Recommendation Algorithms Development
In the development of our recommendation system, we have designed two distinct algorithms. The first algorithm is built on K-Means Clustering, and the second employs an Item-Centered Approach, primarily utilizing Cosine Similarity as its core mechanism. These approaches, derived from our thorough analysis in previous sections, are tailored to leverage the unique attributes of our song dataset and to address specific challenges in music recommendation.

**Model 1: K-Means Clustering Algorithm**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_songs_keans(song_name, data, kmeans_model, scaler, features, top_n=10):
    """
    Recommend top n songs closest to the seed song based on KMeans clustering.

    Args:
        song_name: Name of the song to base recommendations on.
        data: Pandas DataFrame of song data.
        kmeans_model: Trained KMeans model.
        scaler: StandardScaler object used for feature scaling.
        features: List of features used in clustering.
        top_n: Number of top recommendations to return.

    Returns:
        Pandas DataFrame of top n recommended songs closest to the seed song.
    """
    if song_name not in data['name'].values:
        return "Song not found in the dataset."

    # Find the feature vector of the seed song
    seed_song_features = scaler.transform(data[data['name'] == song_name][features])

    # Predict the cluster for the seed song
    song_cluster = kmeans_model.predict(seed_song_features)[0]

    # Filter out the songs from the same cluster
    cluster_songs = data[data['cluster'] == song_cluster]

    # Calculate similarity scores for all songs in the same cluster
    similarities = cosine_similarity(seed_song_features, scaler.transform(cluster_songs[features]))
    cluster_songs['similarity'] = np.squeeze(similarities)

    # Sort the songs based on similarity scores
    recommendations = cluster_songs.sort_values(by='similarity', ascending=False)

    # Exclude the seed song and return the top n songs
    recommendations = recommendations[recommendations['name'] != song_name]
    return recommendations[['name', 'artists', 'year']].head(top_n)
```

The K-Means Clustering algorithm we have developed groups songs into clusters based on their inherent features. This method allows us to recommend songs that are not just similar in a superficial sense but are closely aligned in terms of their underlying characteristics. The steps involved in this recommendation process are as follows:

1. **Feature Vector Extraction:** When a user selects a song, the algorithm first identifies the feature vector of this seed song. This is achieved through the transformation of its features using a pre-fitted StandardScaler, ensuring that the song's features are on the same scale as those used to train the K-Means model.

2. **Cluster Identification:** The algorithm then determines which cluster the seed song belongs to by using the trained K-Means model. This step is crucial as it narrows down the pool of potential recommendations to only those songs that reside in the same cluster.

3. **Cosine Similarity Calculation:** Within the identified cluster, the algorithm calculates the cosine similarity between the seed song and all other songs. This similarity score quantifies how close each song in the cluster is to the seed song in terms of their features.

4. **Sorting and Recommendation:** The songs are then sorted based on their similarity scores. The higher the score, the more similar the song is to the seed song. The algorithm finally presents the top `n` songs as recommendations, excluding the seed song itself.

**Model 2: Item-Centered Content-Based Filtering**
```python
def recommend_songs_content_based(song_name, data, scaler, features, top_n=10, popularity_weight=0.5):
    """
    Recommend top n songs based on a mix of cosine similarity and popularity.

    Args:
        song_name: Name of the seed song to base recommendations on.
        data: Pandas DataFrame of song data.
        scaler: StandardScaler object used for feature scaling.
        features: List of features used in clustering.
        top_n: Number of top recommendations to return.
        popularity_weight: Weight for popularity in the final score (0 to 1).

    Returns:
        Pandas DataFrame of top n recommended songs.
    """
    if song_name not in data['name'].values:
        return "Seed song not found in the dataset."

    # Find the feature vector of the seed song
    song_features = scaler.transform(data[data['name'] == song_name][features])

    # Calculate similarity scores for all songs
    similarities = cosine_similarity(song_features, scaler.transform(data[features]))
    data['similarity'] = np.squeeze(similarities)

    # Normalize popularity
    data['normalized_popularity'] = data['popularity'] / data['popularity'].max()

    # Combine similarity and popularity
    data['score'] = (1 - popularity_weight) * data['similarity'] + popularity_weight * data['normalized_popularity']

    # Sort the songs based on the combined score
    recommendations = data.sort_values(by='score', ascending=False)

    # Exclude the seed song and return the top n songs
    recommendations = recommendations[recommendations['name'] != song_name]
    return recommendations[['name', 'artists', 'year', 'score']].head(top_n)
```

The Item-Centered Content-Based Filtering algorithm developed for song recommendations operates by evaluating the intrinsic features of the songs. It offers recommendations that are deeply aligned with the specific characteristics of the chosen seed song. The key steps in this process are:

1. **Feature Vector Extraction:** The algorithm starts by identifying the feature vector of the selected seed song. It does this by transforming the song's features using a StandardScaler, which ensures consistency in scale with the features used in the model's training.

2. **Cosine Similarity Calculation:** Unlike the K-Means model, this method directly calculates the cosine similarity between the seed song and all other songs in the dataset, measuring how closely they match in terms of their features.

3. **Popularity Integration:** It then incorporates the popularity aspect, normalizing popularity scores, and blending them with the similarity scores. This creates a balanced metric that considers both how similar a song is to the seed song and its overall popularity.

4. **Sorting and Recommendation:** The songs are sorted based on this composite score, and the top n songs, excluding the seed song itself, are presented as recommendations.


In summary, K-Means Clustering Algorithm groups songs into clusters based on their features. Recommendations are made by first identifying the cluster of a selected seed song and then recommending songs within that cluster based on their similarity to the seed song. Item-Centered Content-Based Filtering calculates recommendations by directly comparing the features of the seed song to all songs in the dataset, incorporating popularity into the scoring. This model offers a more direct, feature-focused approach compared to the cluster-based strategy.

## 6. Model Evaluation
Given that our dataset is exclusively song-centric, assessing the performance of our recommendation algorithms presents a unique challenge. Traditional evaluation methods commonly applied to unsupervised learning algorithms are not entirely suitable in this context. To address this, we've opted for industry-standard metrics specifically tailored for evaluating recommendation systems. These metrics include diversity, novelty, and top-k accuracy.

**Diversity** refers to the level of dissimilarity among the recommended items for a user. This dissimilarity can be assessed based on the content of the items, such as differences in music genres, or it can be evaluated based on the variation in how users rate these items [16].

**Novelty** evaluates the freshness of items in a recommendation. It includes two dimensions: user-dependent and user-independent novelty. user-dependent novelty assesses the degree to which the recommendations are distinct or new to a specific user, signifying the introduction of previously undiscovered or untried content. Conversely, user-independent novelty focuses on the overall newness of the recommendations within the entire system, irrespective of individual user familiarity [17]. In our evaluation, we will use user-independent novelty.

**Top-K accuracy** is a vital metric in evaluating recommendation systems, particularly emphasizing the system's precision in predicting the most relevant items and placing them within the top `K` positions of a recommendation list [18]. This metric assumes significance in practical scenarios where users typically consider only the first few recommendations. It assesses the likelihood of the user finding the recommended items genuinely relevant or useful, especially among the top few suggestions [18]. Accurately capturing this aspect is crucial for understanding the practical effectiveness of a recommendation system in real-world user interactions.

## 7. Result and Discussion

## 8. Conclusion and Future Work

## 9. References
[1] Y. E. Ay, "Spotify Dataset 1921-2020, 600k+ Tracks," Kaggle, 2021. Available: https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks. Accessed on: November 21, 2023.
[2] Spotify. "Get Track." Spotify for Developers.  Available: https://developer.spotify.com/documentation/web-api/reference/get-track. Accessed on: November 22, 2023.
[3] Spotify. "Get Several Audio Features." Spotify for Developers. Available: https://developer.spotify.com/documentation/web-api/reference/get-several-audio-features. Accessed on: November 22, 2023.
[4] Maarten Grootendorst. "9 Distance Measures in Data Science." Maarten Grootendorst, https://www.maartengrootendorst.com/blog/distances/. Accessed: December 8, 2023.
[5] "Cosine Similarity." Wikipedia, Wikimedia Foundation, https://en.wikipedia.org/wiki/Cosine_similarity. Accessed: December 8, 2023.
[6] Shkhanukova, Milana. "Cosine Distance and Cosine Similarity.", Medium, https://medium.com/@milana.shxanukova15/cosine-distance-and-cosine-similarity-a5da0e4d9ded. Accessed: December 8, 2023.
[7] Ajay Patel. "Relationship between Cosine Similarity and Euclidean Distance.", https://ajayp.app/posts/2020/05/relationship-between-cosine-similarity-and-euclidean-distance/. Accessed: December 8, 2023.
[8] Rastogi, V. “Euclidean distance and cosine similarity”, Medium, https://medium.com/@vaibhav1403/euclidean-distance-and-cosine-similarity-69cbf8140fed. Accessed: December 8, 2023.
[9] "Cluster analysis," Wikipedia, The Free Encyclopedia, [Online]. Available: https://en.wikipedia.org/wiki/Cluster_analysis. Accessed: December 9, 2023.
[10] G. Learning, "Clustering algorithms," Medium, https://medium.com/@mygreatlearning/clustering-algorithms-d7b3ae040a95. Accessed: December 9, 2023.
[11] N. Sharma, "K-means clustering explained," neptune.ai, https://neptune.ai/blog/k-means-clustering Accessed: December 9, 2023.
[12] T. Firdose, "Understanding the Elbow Method: Finding the Optimal Number of Clusters," Medium, Available: https://tahera-firdose.medium.com/understanding-the-elbow-method-finding-the-optimal-number-of-clusters-68319d773ea3. Accessed: December 10, 2023.
[13] "Principal component analysis," Wikipedia, The Free Encyclopedia, Available: https://en.wikipedia.org/wiki/Principal_component_analysis. Accessed: December 11, 2023.
[14] "Dimensionality reduction: PCA, tsne, umap," Auriga IT, https://aurigait.com/blog/blog-easy-explanation-of-dimensionality-reduction-and-techniques/ Accessed: December 11, 2023.
[15] Rocca, Baptiste. "Introduction to Recommender Systems." Medium, Towards Data Science, https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada Accessed: December 10, 2023.
[16] B, Anna. "Recommender Systems - It’s Not All about the Accuracy." Medium, https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff. 
Accessed: December 11, 2023.
[17] Pabalan, Christabelle. "Beyond Accuracy: Embracing Serendipity and Novelty in Recommendations for Long Term User Retention." Towards Data Science, https://towardsdatascience.com/beyond-accuracy-embracing-serendipity-and-novelty-in-recommendations-for-long-term-user-retention-701a23b1cb34. Accessed: December 11, 2023.
[18] S. Kapre, "Common metrics to evaluate recommendation systems," Medium, [Online]. Available: https://flowthytensor.medium.com/some-metrics-to-evaluate-recommendation-systems-9e0cf0c8b6cf. Accessed: December 11, 2023.


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


