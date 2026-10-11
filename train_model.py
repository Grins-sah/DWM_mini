
from pathlib import Path
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ml-1m"
ARTIFACT_DIR = BASE_DIR / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1"
    )

    movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )

    return ratings, movies


def build_user_profiles(ratings, movies):
    # One row per user-movie interaction with movie genres.
    data = ratings.merge(
        movies[["movieId", "genres"]],
        on="movieId",
        how="inner"
    )

    data["genre"] = data["genres"].str.split("|")
    data = data.explode("genre")

    # Average rating given by each user to each genre.
    profiles = data.pivot_table(
        index="userId",
        columns="genre",
        values="rating",
        aggfunc="mean"
    )

    profiles.columns.name = None

    # Use the user's average rating for unseen genres.
    user_means = ratings.groupby("userId")["rating"].mean()
    profiles = profiles.T.fillna(user_means).T

    return profiles, data


def train():
    print("Loading MovieLens data...")
    ratings, movies = load_data()

    print("Building user preference vectors...")
    profiles, expanded_data = build_user_profiles(
        ratings, movies
    )

    # Standardize genre preference features.
    scaler = StandardScaler()
    X = scaler.fit_transform(profiles)

    # Mandatory clustering algorithm: K-Means.
    model = KMeans(
        n_clusters=8,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    # Evaluate clustering quality.
    if len(set(labels)) > 1:
        sample_size = min(10000, len(X))

        score = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=42
        )
        print(f"Silhouette score: {score:.4f}")

    # Map every historical user to a cluster.
    user_clusters = pd.DataFrame({
        "userId": profiles.index,
        "cluster": labels
    })

    # Interpretable cluster profiles in original rating scale.
    profile_with_cluster = profiles.copy()
    profile_with_cluster["cluster"] = labels

    cluster_profiles = (
        profile_with_cluster
        .groupby("cluster")
        .mean()
    )

    # Average rating for each movie by users in each cluster.
    rated_users = ratings.merge(
        user_clusters,
        on="userId",
        how="inner"
    )

    cluster_movie_scores = (
        rated_users
        .groupby(["cluster", "movieId"])
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "size")
        )
        .reset_index()
    )

    # Movie catalogue with genre lists.
    movies = movies.copy()
    movies["genre_list"] = movies["genres"].apply(
        lambda value: [
            g for g in value.split("|") if g != "(no genres listed)"
        ]
    )

    # Save all artifacts required by the backend.
    artifacts = {
        "model": model,
        "scaler": scaler,
        "genres": list(profiles.columns),
        "user_clusters": user_clusters,
        "cluster_profiles": cluster_profiles,
        "cluster_movie_scores": cluster_movie_scores,
        "movies": movies,
        "user_profiles": profiles
    }

    output_file = ARTIFACT_DIR / "movie_model.joblib"
    joblib.dump(artifacts, output_file)

    print("\nTraining complete!")
    print(f"Users: {len(profiles)}")
    print(f"Movies: {len(movies)}")
    print(f"Ratings: {len(ratings)}")
    print(f"Clusters: {len(set(labels))}")
    print(f"Saved model: {output_file}")

    print("\nCluster sizes:")
    print(user_clusters["cluster"].value_counts().sort_index())

    print("\nCluster genre profiles:")
    for cluster_id, row in cluster_profiles.iterrows():
        top_genres = row.sort_values(ascending=False).head(5)
        print(
            f"Cluster {cluster_id}: "
            + ", ".join(
                f"{genre} ({rating:.2f})"
                for genre, rating in top_genres.items()
            )
        )


if __name__ == "__main__":
    train()
