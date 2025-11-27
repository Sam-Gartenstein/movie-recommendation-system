from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd

from recommender.item_item_cf import recommend_similar, recommend_for_user


# ---------- Pydantic response models ----------

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    score: float

        
class HealthStatus(BaseModel):
    status: str
    movies_loaded: bool
    user_item_loaded: bool
    similarity_loaded: bool


app = FastAPI(title="Movie Recommender (Item–Item CF)")

# ---------- Global artifacts (lazy-loaded) ----------

movies_df: Optional[pd.DataFrame] = None
user_item_matrix: Optional[pd.DataFrame] = None
item_similarity_df: Optional[pd.DataFrame] = None


@app.on_event("startup")
def load_artifacts() -> None:
    """
    Load model artifacts once at startup.
    If something fails here, the /health endpoint will show it.
    """
    global movies_df, user_item_matrix, item_similarity_df

    try:
        movies_df = joblib.load("artifacts/movies_df.pkl")
    except Exception:
        movies_df = None

    try:
        user_item_matrix = joblib.load("artifacts/user_item_matrix.pkl")
    except Exception:
        user_item_matrix = None

    try:
        item_similarity_df = joblib.load("artifacts/item_similarity_df.pkl")
    except Exception:
        item_similarity_df = None


# ---------- Utility ----------

def ensure_artifacts_loaded(require_user_data: bool = False) -> None:
    """
    Ensure that required artifacts are in memory before serving a request.
    """
    if movies_df is None or item_similarity_df is None:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts are not loaded. Check server logs.",
        )
    if require_user_data and user_item_matrix is None:
        raise HTTPException(
            status_code=500,
            detail="User–item matrix not loaded. Check server logs.",
        )


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    """
    Simple health check for the service.
    """
    return HealthStatus(
        status="ok",
        movies_loaded=movies_df is not None,
        user_item_loaded=user_item_matrix is not None,
        similarity_loaded=item_similarity_df is not None,
    )


@app.get("/similar", response_model=List[MovieRecommendation])
def similar(movie: str, n: int = 10):
    """
    Recommend movies similar to a given movie title using item–item CF.

    Query params:
      - movie: movie title (string)
      - n: number of recommendations to return (default: 10)
    """
    ensure_artifacts_loaded(require_user_data=False)

    try:
        recs = recommend_similar(
            movie=movie,
            movies_df=movies_df,
            item_similarity_df=item_similarity_df,
            n=n,
        )
    except ValueError as e:
        # e.g. movie not found
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Make sure the columns match MovieRecommendation fields
    return [
        MovieRecommendation(
            movie_id=int(row["movie_id"]),
            title=str(row["title"]),
            score=float(row["score"]),
        )
        for _, row in recs.iterrows()
    ]


@app.get("/recommend", response_model=List[MovieRecommendation])
def recommend(user_id: int, n: int = 10):
    """
    Recommend movies for a given user using item–item CF.

    Query params:
      - user_id: integer user id
      - n: number of recommendations to return (default: 10)
    """
    ensure_artifacts_loaded(require_user_data=True)

    try:
        recs = recommend_for_user(
            user_id=user_id,
            movies_df=movies_df,
            user_item_matrix=user_item_matrix,
            item_similarity_df=item_similarity_df,
            n_recs=n,
        )
        if recs is None or recs.empty:
            raise HTTPException(
                status_code=404,
                detail="No recommendations available for this user.",
            )
    except ValueError as e:
        # e.g. user not found
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return [
        MovieRecommendation(
            movie_id=int(row["movie_id"]),
            title=str(row["title"]),
            score=float(row["score"]),
        )
        for _, row in recs.iterrows()
    ]

