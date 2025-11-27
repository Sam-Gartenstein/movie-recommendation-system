from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np

from recommender.item_item_cf import recommend_similar, recommend_for_user

app = FastAPI(title="Movie Recommender (Item–Item CF)")

# Load artifacts at startup
movies_df = joblib.load("artifacts/movies_df.pkl")
user_item_matrix = joblib.load("artifacts/user_item_matrix.pkl")
item_similarity_df = joblib.load("artifacts/item_similarity_df.pkl")


@app.get("/similar")
def similar(movie: str, n: int = 10):
    """
    Recommend movies similar to a given movie title using item–item CF.
    """
    try:
        recs = recommend_similar(
            movie=movie,
            movies_df=movies_df,
            item_similarity_df=item_similarity_df,
            n=n,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return recs.to_dict(orient="records")


@app.get("/recommend")
def recommend(user_id: int, n: int = 10):
    """
    Recommend movies for a given user using item–item CF.
    """
    try:
        recs = recommend_for_user(
            user_id=user_id,
            movies_df=movies_df,
            user_item_matrix=user_item_matrix,
            item_similarity_df=item_similarity_df,
            n_recs=n,
        )
        if recs is None:
            raise HTTPException(status_code=404, detail="No recommendations available.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return recs.to_dict(orient="records")

