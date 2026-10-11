# ReelCluster frontend

React + Vite frontend for the FastAPI cluster-based movie recommender.

## Requirements
- Node.js 18+ (Node 20+ recommended)
- FastAPI backend running with `artifacts/movie_model.joblib`

## Run

```bash
npm install
npm run dev
```

Open the local URL Vite prints (normally `http://localhost:5173`).

Default API URL is `http://127.0.0.1:8000`. Use **API settings** in the sidebar to change it and test the connection.

## Features
- New-user onboarding with genre sliders
- Existing-user profile loading and preference updates
- K-Means cluster display
- Movie recommendations and like/dislike/watchlist interactions
- Add a new movie and generate cluster-targeted in-app notifications
- Notifications inbox and mark-as-read
- Local catalogue view

## Backend note
The FastAPI backend must allow browser CORS requests from the Vite development server. Add this to `app/main.py` after creating `app = FastAPI(...)` if not already configured:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Restart the API after editing it.

## Data note
Movie cards use generated graphic poster art; the backend returns MovieLens titles, genres, and rating aggregates, not official poster images. The "My catalogue" tab lists movies imported into or added to the backend's SQLite database.
