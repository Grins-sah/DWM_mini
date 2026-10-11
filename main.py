import joblib
import numpy as np

artifacts = joblib.load(
    "artifacts/movie_model.joblib"
)

def assign_user_to_cluster(preferences):
    """
    preferences:
    {
        "Action": 5,
        "Comedy": 4,
        "Drama": 2,
        "Horror": 1
    }

    Unspecified genres default to a neutral score of 3.
    """
    genres = artifacts["genres"]

    values = np.array([
        preferences.get(genre, 3.0)
        for genre in genres
    ]).reshape(1, -1)

    scaled = artifacts["scaler"].transform(values)

    cluster = int(
        artifacts["model"].predict(scaled)[0]
    )

    return cluster
def recommend_movies(cluster_id, limit=10):
    scores = artifacts["cluster_movie_scores"]

    movies = artifacts["movies"]

    candidates = scores[
        (scores["cluster"] == cluster_id)
        & (scores["rating_count"] >= 3)
    ]

    # Avoid recommending movies already rated by this user
    # by filtering their movie IDs in your application layer.

    recommendations = candidates.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    return (
        recommendations
        .sort_values(
            ["avg_rating", "rating_count"],
            ascending=[False, False]
        )
        .head(limit)[
            ["movieId", "title", "genres",
             "avg_rating", "rating_count"]
        ]
    )
if __name__ == "__main__":
    preferences = {
        "Action": 5,
        "Adventure": 5,
        "Sci-Fi": 5,
        "Comedy": 2,
        "Romance": 5,
        "Horror": 5
    }

    cluster_id = assign_user_to_cluster(preferences)
    print("Assigned cluster:", cluster_id)
    movies = recommend_movies(cluster_id)
    print(movies)