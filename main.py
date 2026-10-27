from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = 'A:/DWM/artifacts/movie_model.joblib'
DB_PATH = 'A:/DWM/movie_recommender.db'

app = FastAPI(
    title="Cluster-Based Movie Recommendation API",
    description=(
        "K-Means user clustering, movie recommendations, user interactions, "
        "and cluster-targeted notifications."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

artifacts: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db_connection() as conn:
        conn.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            cluster_id INTEGER NOT NULL,
            preferences_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS movies (
            movie_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            genres_json TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            is_catalogue_movie INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            movie_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            rating REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_interactions_user
        ON interactions(user_id);

        CREATE INDEX IF NOT EXISTS idx_interactions_movie
        ON interactions(movie_id);

        CREATE TABLE IF NOT EXISTS notifications (
            notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            movie_id TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            read INTEGER NOT NULL DEFAULT 0,
            UNIQUE(user_id, movie_id),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS cluster_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            movie_id TEXT,
            movie_title TEXT,
            event_type TEXT,
            cluster_before INTEGER,
            cluster_after INTEGER NOT NULL,
            preferences_before_json TEXT NOT NULL,
            preferences_after_json TEXT NOT NULL,
            genre_effect_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cluster_history_user
        ON cluster_history(user_id, history_id DESC);
        """)


@app.on_event("startup")
def startup() -> None:
    global artifacts
    init_db()

    artifacts = joblib.load(MODEL_PATH)


def require_model() -> dict[str, Any]:
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model artifact not found or not loaded at {MODEL_PATH}. "
                "Run train_model.py first, then place artifacts/movie_model.joblib "
                "in the backend project."
            ),
        )
    required = {
        "model", "scaler", "genres", "user_clusters",
        "cluster_profiles", "cluster_movie_scores", "movies", "user_profiles"
    }
    missing = required.difference(artifacts.keys())
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Model artifact is missing keys: {sorted(missing)}. Retrain using the supplied training script.",
        )
    return artifacts


def json_list(value: Any) -> list[str]:
    import json
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (TypeError, ValueError):
        return []


def normalise_preferences(preferences: dict[str, float]) -> dict[str, float]:
    model_artifacts = require_model()
    valid_genres = set(model_artifacts["genres"])
    cleaned: dict[str, float] = {}
    for genre, score in preferences.items():
        if genre not in valid_genres:
            continue
        if not 1 <= float(score) <= 5:
            raise HTTPException(
                status_code=422,
                detail=f"Preference for '{genre}' must be between 1 and 5.",
            )
        cleaned[genre] = float(score)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail=f"Provide at least one valid genre preference. Valid genres: {sorted(valid_genres)}",
        )
    return cleaned


def assign_cluster(preferences: dict[str, float]) -> int:
    model_artifacts = require_model()
    genres = model_artifacts["genres"]
    # The training code filled missing genre ratings with each user's mean.
    # For questionnaire-only users, use the mean of supplied preferences.
    neutral = float(np.mean(list(preferences.values())))
    values = np.array([preferences.get(g, neutral) for g in genres]).reshape(1, -1)
    scaled = model_artifacts["scaler"].transform(values)
    return int(model_artifacts["model"].predict(scaled)[0])


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class UserCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    preferences: dict[str, float] = Field(
        description="Genre preference scores from 1 to 5, e.g. {'Action': 5, 'Comedy': 3}"
    )


class InteractionCreate(BaseModel):
    movie_id: str = Field(min_length=1, max_length=100)
    event_type: str = Field(description="One of: view, like, dislike, rejected, watchlist, watched, rating")
    rating: float | None = Field(default=None, ge=0.5, le=5.0)


class MovieCreate(BaseModel):
    movie_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    genres: list[str] = Field(min_length=1)
    description: str | None = None
    min_cluster_score: float = Field(default=3.5, ge=1, le=5)
    max_clusters: int = Field(default=3, ge=1, le=100)


class PreferencesUpdate(BaseModel):
    preferences: dict[str, float]


class NotificationReadUpdate(BaseModel):
    read: bool = True


@app.get("/")
def root():
    return {
        "message": "Cluster-Based Movie Recommendation API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": artifacts is not None,
        "model_path": str(MODEL_PATH),
        "database_path": str(DB_PATH),
    }


@app.get("/genres")
def get_genres():
    model_artifacts = require_model()
    return {"genres": model_artifacts["genres"]}


@app.post("/users", status_code=201)
def create_user(payload: UserCreate):
    """Create a new user from onboarding questionnaire answers."""
    model_artifacts = require_model()
    preferences = normalise_preferences(payload.preferences)
    cluster_id = assign_cluster(preferences)

    import json
    now = utc_now()
    with db_connection() as conn:
        existing = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?", (payload.user_id,)
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="User ID already exists. Use PUT /users/{user_id} to update preferences.")

        conn.execute(
            """INSERT INTO users
            (user_id, cluster_id, preferences_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (payload.user_id, cluster_id, json.dumps(preferences), now, now),
        )
        conn.execute(
            """INSERT INTO cluster_history
            (user_id, action_type, event_type, cluster_before, cluster_after,
             preferences_before_json, preferences_after_json, genre_effect_json, created_at)
            VALUES (?, 'onboarding', 'onboarding', NULL, ?, '{}', ?, '{}', ?)""",
            (payload.user_id, cluster_id, json.dumps(preferences), now),
        )

    return {
        "user_id": payload.user_id,
        "cluster_id": cluster_id,
        "preferences": preferences,
        "message": "User created and assigned to a K-Means cluster.",
        "recommendations": _recommend_for_cluster(cluster_id, limit=10, exclude_movie_ids=set()),
    }


@app.get("/users/{user_id}")
def get_user(user_id: str):
    """Retrieve an existing user's cluster and preferences."""
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found.")
    import json
    user = dict(row)
    user["preferences"] = json.loads(user.pop("preferences_json"))
    return user


@app.put("/users/{user_id}")
def update_user_preferences(user_id: str, payload: PreferencesUpdate):
    """Update questionnaire preferences, preserve the before/after audit, and reassign cluster."""
    import json
    preferences = normalise_preferences(payload.preferences)
    cluster_id = assign_cluster(preferences)
    now = utc_now()
    with db_connection() as conn:
        existing = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="User not found.")
        before_preferences = json.loads(existing["preferences_json"])
        before_cluster = int(existing["cluster_id"])
        conn.execute(
            """UPDATE users SET cluster_id = ?, preferences_json = ?, updated_at = ? WHERE user_id = ?""",
            (cluster_id, json.dumps(preferences), now, user_id),
        )
        conn.execute(
            """INSERT INTO cluster_history
            (user_id, action_type, event_type, cluster_before, cluster_after,
             preferences_before_json, preferences_after_json, genre_effect_json, created_at)
            VALUES (?, 'preferences_updated', 'preferences_updated', ?, ?, ?, ?, ?, ?)""",
            (user_id, before_cluster, cluster_id, json.dumps(before_preferences), json.dumps(preferences),
             json.dumps({g: round(preferences.get(g, 3) - before_preferences.get(g, 3), 3)
                         for g in set(before_preferences) | set(preferences)}), now),
        )
    return {
        "user_id": user_id,
        "cluster_id": cluster_id,
        "previous_cluster_id": before_cluster,
        "preferences": preferences,
        "message": "Preferences updated and cluster reassigned.",
    }


@app.post("/users/{user_id}/interactions")
def record_interaction(user_id: str, payload: InteractionCreate):
    """Record an interaction, apply a small genre-affinity update, and save an explainable audit row."""
    import json
    allowed = {"view", "like", "dislike", "rejected", "watchlist", "watched", "rating"}
    if payload.event_type not in allowed:
        raise HTTPException(status_code=422, detail=f"event_type must be one of {sorted(allowed)}")
    model_artifacts = require_model()
    valid_genres = set(model_artifacts["genres"])
    now = utc_now()

    with db_connection() as conn:
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        movie = conn.execute("SELECT * FROM movies WHERE movie_id = ?", (payload.movie_id,)).fetchone()
        if movie:
            title = str(movie["title"])
            movie_genres = json_list(movie["genres_json"])
        else:
            catalogue = model_artifacts["movies"]
            matches = catalogue[catalogue["movieId"].astype(str) == payload.movie_id]
            if matches.empty:
                raise HTTPException(status_code=404, detail="Movie not found in local database or MovieLens catalogue.")
            match = matches.iloc[0]
            title = str(match["title"])
            movie_genres = [g for g in str(match["genres"]).split("|") if g and g != "(no genres listed)"]
            conn.execute(
                """INSERT OR IGNORE INTO movies
                (movie_id, title, genres_json, description, created_at, is_catalogue_movie)
                VALUES (?, ?, ?, NULL, ?, 1)""",
                (payload.movie_id, title, json.dumps(movie_genres), now),
            )

        before_preferences = json.loads(user["preferences_json"])
        before_cluster = int(user["cluster_id"])
        after_preferences = dict(before_preferences)
        genre_effect: dict[str, float] = {}
        relevant_genres = [g for g in movie_genres if g in valid_genres]

        # Implicit feedback is deliberately small so a single click does not dominate the profile.
        if payload.event_type in {"watched", "rejected", "dislike", "like", "rating"}:
            for genre in relevant_genres:
                old_value = float(after_preferences.get(genre, 3.0))
                if payload.event_type == "rating" and payload.rating is not None:
                    new_value = 0.7 * old_value + 0.3 * float(payload.rating)
                elif payload.event_type == "watched":
                    new_value = old_value + 0.25
                elif payload.event_type == "like":
                    new_value = old_value + 0.50
                else:  # rejected / dislike
                    new_value = old_value - 0.75
                new_value = round(min(5.0, max(1.0, new_value)), 3)
                after_preferences[genre] = new_value
                genre_effect[genre] = round(new_value - old_value, 3)

        after_cluster = assign_cluster(after_preferences) if genre_effect else before_cluster
        conn.execute(
            """INSERT INTO interactions (user_id, movie_id, event_type, rating, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (user_id, payload.movie_id, payload.event_type, payload.rating, now),
        )
        if genre_effect:
            conn.execute(
                """UPDATE users SET preferences_json = ?, cluster_id = ?, updated_at = ? WHERE user_id = ?""",
                (json.dumps(after_preferences), after_cluster, now, user_id),
            )
        conn.execute(
            """INSERT INTO cluster_history
            (user_id, action_type, movie_id, movie_title, event_type, cluster_before, cluster_after,
             preferences_before_json, preferences_after_json, genre_effect_json, created_at)
            VALUES (?, 'movie_interaction', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, payload.movie_id, title, payload.event_type, before_cluster, after_cluster,
             json.dumps(before_preferences), json.dumps(after_preferences), json.dumps(genre_effect), now),
        )

    return {
        "user_id": user_id,
        "movie_id": payload.movie_id,
        "movie_title": title,
        "genres": movie_genres,
        "event_type": payload.event_type,
        "rating": payload.rating,
        "cluster_before": before_cluster,
        "cluster_after": after_cluster,
        "cluster_changed": before_cluster != after_cluster,
        "preferences_before": before_preferences,
        "preferences_after": after_preferences,
        "genre_effect": genre_effect,
        "message": "Interaction recorded and taste profile recalculated." if genre_effect else "Interaction recorded; no taste-profile change for this event type.",
    }


def _recommend_for_cluster(
    cluster_id: int,
    limit: int,
    exclude_movie_ids: set[str],
) -> list[dict[str, Any]]:
    model_artifacts = require_model()
    scores = model_artifacts["cluster_movie_scores"]
    catalogue = model_artifacts["movies"]

    candidates = scores[
        (scores["cluster"] == cluster_id)
        & (scores["rating_count"] >= 3)
    ].copy()
    candidates["movieId"] = candidates["movieId"].astype(str)
    if exclude_movie_ids:
        candidates = candidates[~candidates["movieId"].isin(exclude_movie_ids)]

    # Penalize very low-confidence items and favor broadly well-rated movies.
    candidates["confidence_adjusted_score"] = (
        candidates["avg_rating"] * np.log1p(candidates["rating_count"])
    )
    catalogue_for_merge = catalogue[["movieId", "title", "genres"]].copy()
    catalogue_for_merge["movieId"] = catalogue_for_merge["movieId"].astype(str)
    merged = candidates.merge(
        catalogue_for_merge,
        on="movieId",
        how="left",
    )
    merged = merged.sort_values(
        ["confidence_adjusted_score", "avg_rating"],
        ascending=False,
    ).head(limit)

    return [
        {
            "movie_id": str(row.movieId),
            "title": str(row.title),
            "genres": [
                g for g in str(row.genres).split("|")
                if g != "(no genres listed)"
            ],
            "average_cluster_rating": round(float(row.avg_rating), 3),
            "rating_count_in_cluster": int(row.rating_count),
        }
        for row in merged.itertuples(index=False)
    ]


@app.get("/users/{user_id}/recommendations")
def recommend_for_user(user_id: str, limit: int = 10):
    """Recommend historical movies for an existing user, excluding interacted movies."""
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100.")

    with db_connection() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        interacted = conn.execute(
            "SELECT DISTINCT movie_id FROM interactions WHERE user_id = ?",
            (user_id,),
        ).fetchall()

    excluded = {str(row["movie_id"]) for row in interacted}
    recommendations = _recommend_for_cluster(
        int(user["cluster_id"]), limit, excluded
    )
    return {
        "user_id": user_id,
        "cluster_id": int(user["cluster_id"]),
        "recommendations": recommendations,
    }


@app.get("/movies")
def list_movies(limit: int = 50, q: str | None = None):
    """List movies stored in the app database; optionally filter by title."""
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200.")
    with db_connection() as conn:
        if q:
            rows = conn.execute(
                """SELECT * FROM movies WHERE title LIKE ?
                ORDER BY title LIMIT ?""",
                (f"%{q}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM movies ORDER BY title LIMIT ?", (limit,)
            ).fetchall()
    return [
        {
            "movie_id": row["movie_id"],
            "title": row["title"],
            "genres": json_list(row["genres_json"]),
            "description": row["description"],
            "is_catalogue_movie": bool(row["is_catalogue_movie"]),
        }
        for row in rows
    ]


@app.post("/movies", status_code=201)
def add_new_movie(payload: MovieCreate):
    """
    Add a new movie and create notifications for the clusters whose genre
    preference score passes min_cluster_score. This endpoint does not send
    email/push messages; it stores in-app notification records.
    """
    model_artifacts = require_model()
    valid_genres = set(model_artifacts["genres"])
    genres = list(dict.fromkeys(payload.genres))
    known_genres = [genre for genre in genres if genre in valid_genres]
    if not known_genres:
        raise HTTPException(
            status_code=422,
            detail=f"At least one supplied genre must match the model's genres: {sorted(valid_genres)}",
        )

    cluster_scores = []
    profiles = model_artifacts["cluster_profiles"]
    for cluster_id, profile in profiles.iterrows():
        score = float(profile[known_genres].mean())
        cluster_scores.append((int(cluster_id), score))

    cluster_scores.sort(key=lambda item: item[1], reverse=True)
    target_clusters = [
        (cluster_id, score)
        for cluster_id, score in cluster_scores
        if score >= payload.min_cluster_score
    ][:payload.max_clusters]

    import json
    now = utc_now()
    created_notifications = 0
    with db_connection() as conn:
        exists = conn.execute(
            "SELECT movie_id FROM movies WHERE movie_id = ?",
            (payload.movie_id,),
        ).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="Movie ID already exists.")

        conn.execute(
            """INSERT INTO movies
            (movie_id, title, genres_json, description, created_at, is_catalogue_movie)
            VALUES (?, ?, ?, ?, ?, 0)""",
            (
                payload.movie_id,
                payload.title,
                json.dumps(genres),
                payload.description,
                now,
            ),
        )

        for cluster_id, score in target_clusters:
            users = conn.execute(
                "SELECT user_id FROM users WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchall()
            for user in users:
                message = (
                    f"New movie '{payload.title}' may match your interests "
                    f"(cluster {cluster_id}, genre match score {score:.2f}/5)."
                )
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO notifications
                    (user_id, movie_id, message, created_at)
                    VALUES (?, ?, ?, ?)""",
                    (user["user_id"], payload.movie_id, message, now),
                )
                created_notifications += max(cursor.rowcount, 0)

    return {
        "movie_id": payload.movie_id,
        "title": payload.title,
        "genres": genres,
        "target_clusters": [
            {"cluster_id": cluster_id, "genre_preference_score": round(score, 3)}
            for cluster_id, score in target_clusters
        ],
        "notifications_created": created_notifications,
        "note": "Notifications are saved in the app database; connect an email/push provider to deliver them externally.",
    }


@app.get("/users/{user_id}/notifications")
def get_notifications(user_id: str, unread_only: bool = False):
    with db_connection() as conn:
        user = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        if unread_only:
            rows = conn.execute(
                """SELECT * FROM notifications
                WHERE user_id = ? AND read = 0
                ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM notifications
                WHERE user_id = ?
                ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()

    return {"user_id": user_id, "notifications": [dict(row) for row in rows]}


@app.patch("/notifications/{notification_id}")
def mark_notification_read(
    notification_id: int,
    payload: NotificationReadUpdate,
):
    with db_connection() as conn:
        cursor = conn.execute(
            "UPDATE notifications SET read = ? WHERE notification_id = ?",
            (int(payload.read), notification_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Notification not found.")
    return {"notification_id": notification_id, "read": payload.read}


@app.get("/users/{user_id}/cluster-explanation")
def explain_user_cluster(user_id: str):
    """Explain the user's current K-Means assignment using their feature vector and centroid distances."""
    import json
    model_artifacts = require_model()
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found.")
    preferences = json.loads(row["preferences_json"])
    genres = list(model_artifacts["genres"])
    neutral = float(np.mean(list(preferences.values()))) if preferences else 3.0
    feature_vector = np.array([preferences.get(g, neutral) for g in genres], dtype=float).reshape(1, -1)
    scaled = model_artifacts["scaler"].transform(feature_vector)
    centers = model_artifacts["model"].cluster_centers_
    distances = np.linalg.norm(centers - scaled, axis=1)
    closest = sorted(
        [{"cluster_id": int(i), "distance": round(float(d), 5)} for i, d in enumerate(distances)],
        key=lambda item: item["distance"],
    )
    profiles = model_artifacts["cluster_profiles"]
    profile = profiles.loc[int(row["cluster_id"])] if int(row["cluster_id"]) in profiles.index else None
    top_genres = []
    if profile is not None:
        top_genres = [{"genre": str(g), "average_rating": round(float(v), 3)}
                      for g, v in profile.sort_values(ascending=False).head(8).items()]
    return {
        "user_id": user_id,
        "cluster_id": int(row["cluster_id"]),
        "algorithm": "K-Means",
        "feature_scale": "1-5; missing onboarding genres are imputed with the user's mean preference",
        "neutral_imputation_value": round(neutral, 3),
        "feature_vector": [{"genre": g, "value": round(float(feature_vector[0, i]), 3),
                            "provided_by_user": g in preferences}
                           for i, g in enumerate(genres)],
        "cluster_top_genres": top_genres,
        "closest_clusters": closest[:min(5, len(closest))],
        "interpretation": "The assigned cluster is the nearest learned K-Means centroid in scaled genre-feature space. A smaller distance means a closer match; the cluster ID itself is a label, not a ranking.",
    }


@app.get("/users/{user_id}/history")
def get_user_cluster_history(user_id: str, limit: int = 100):
    """Return onboarding, preference edits, and each interaction's before/after cluster and genre values."""
    import json
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500.")
    with db_connection() as conn:
        exists = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="User not found.")
        rows = conn.execute(
            "SELECT * FROM cluster_history WHERE user_id = ? ORDER BY history_id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    history = []
    for row in rows:
        item = dict(row)
        for column, target in [("preferences_before_json", "preferences_before"),
                               ("preferences_after_json", "preferences_after"),
                               ("genre_effect_json", "genre_effect")]:
            try:
                item[target] = json.loads(item.pop(column) or "{}")
            except (TypeError, ValueError):
                item[target] = {}
        item["cluster_changed"] = item.get("cluster_before") is not None and item.get("cluster_before") != item.get("cluster_after")
        history.append(item)
    return {"user_id": user_id, "count": len(history), "history": history}


@app.get("/clusters")
def get_cluster_profiles():
    """Return interpretable genre preference profiles for all clusters."""
    model_artifacts = require_model()
    profiles = model_artifacts["cluster_profiles"]
    result = []
    for cluster_id, row in profiles.iterrows():
        top = row.sort_values(ascending=False).head(5)
        result.append({
            "cluster_id": int(cluster_id),
            "top_genres": [
                {"genre": genre, "average_rating": round(float(score), 3)}
                for genre, score in top.items()
            ],
        })
    return {"clusters": result}
