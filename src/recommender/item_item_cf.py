import numpy as np
import pandas as pd

def recommend_similar(movie, movies_df, item_similarity_df, n=10):
    """
    Recommend movies similar to a given movie.

    Parameters
    ----------
    movie : int or str
        If int, treated as movieId.
        If str, treated as movie title (case-insensitive exact match).
    movies_df : DataFrame
        Movie metadata with at least 'movieId' and 'title' columns.
    item_similarity_df : DataFrame
        Item–item similarity matrix indexed by movieId.
    n : int
        Number of similar movies to return.
    """
    # --- Resolve input to a movieId ---
    if isinstance(movie, int):
        movie_id = movie
    elif isinstance(movie, str):
        # Case-insensitive match on title
        matches = movies_df[movies_df["title"].str.lower() == movie.lower()]
        if matches.empty:
            raise ValueError(f"No movie found with title '{movie}'.")
        movie_id = int(matches.iloc[0]["movieId"])
    else:
        raise TypeError("movie must be an int (movieId) or str (title).")

    # Make sure movieId exists in similarity matrix
    if movie_id not in item_similarity_df.index:
        raise ValueError(f"movie_id {movie_id} not found in similarity matrix.")
    
    # All similarity scores for this movie
    sims = item_similarity_df.loc[movie_id]
    
    # Drop itself
    sims = sims.drop(index=movie_id)
    
    # Sort by similarity descending and take top n ids
    top_ids = sims.sort_values(ascending=False).head(n).index
    
    # Build result with metadata + similarity scores
    result = (
        movies_df[movies_df["movieId"].isin(top_ids)]
        .set_index("movieId")
        .loc[top_ids]  # keep same order as similarity ranking
        .reset_index()
    )
    result["similarity"] = sims.loc[top_ids].values
    
    return result[["movieId", "title", "year", "genres", "similarity"]]


def recommend_for_user(
    user_id,
    movies_df,
    user_item_matrix,
    item_similarity_df,
    n_recs=10,
    min_rated=5,
    neighbor_k=30,
    include_history=False,
    history_n=10,
):
    """
    Recommend movies for a given user_id using item–item CF.

    Parameters
    ----------
    user_id : int
        ID of the user to recommend for.
    movies_df : DataFrame
        Movie metadata with at least 'movieId' and 'title'.
    user_item_matrix : DataFrame
        User–item rating matrix (rows = users, columns = movies).
    item_similarity_df : DataFrame
        Item–item similarity matrix indexed by movieId.
    n_recs : int
        How many recommendations to return.
    min_rated : int
        Warn if user has rated fewer than this many movies.
    neighbor_k : int or None
        Use only top-k most similar rated movies when scoring each candidate.
    include_history : bool
        If True, also return a DataFrame of movies the user has already rated.
    history_n : int
        Number of previously rated movies to show (sorted by rating).
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"user_id {user_id} not found in user_item_matrix.")
    
    user_ratings = user_item_matrix.loc[user_id]
    
    # Movies the user has rated
    rated = user_ratings[user_ratings > 0]
    rated_ids = rated.index.tolist()
    
    if len(rated_ids) < min_rated:
        print(
            f"Warning: user {user_id} has rated only {len(rated_ids)} movies; "
            "recommendations may be noisy."
        )
    
    # Candidate movies: not yet rated
    all_movie_ids = user_item_matrix.columns
    candidate_ids = [m for m in all_movie_ids if m not in rated_ids]
    
    scores = {}
    
    for m in candidate_ids:
        # Similarities between candidate m and movies the user has rated
        sims = item_similarity_df.loc[m, rated_ids].values
        ratings = rated.values
        
        # Use only top-k neighbors (optional but common)
        if neighbor_k is not None and len(sims) > neighbor_k:
            idx = np.argsort(sims)[-neighbor_k:]  # largest k
            sims = sims[idx]
            ratings = ratings[idx]
        
        if np.all(sims == 0):
            continue  # no signal
        
        # Predicted rating (weighted average)
        score = np.dot(sims, ratings) / np.sum(np.abs(sims))
        scores[m] = score
    
    if not scores:
        print("No candidate scores could be computed.")
        return None if not include_history else (None, None)
    
    # Sort candidates by predicted score
    ranked_ids = [
        mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ][:n_recs]
    
    recs = (
        movies_df[movies_df["movieId"].isin(ranked_ids)]
        .set_index("movieId")
        .loc[ranked_ids]
        .reset_index()
    )
    recs["predicted_rating"] = [scores[mid] for mid in ranked_ids]
    recs = recs[["movieId", "title", "year", "genres", "predicted_rating"]]
    
    # Optional: build a small "history" table of movies this user has rated
    if include_history:
        history = (
            movies_df[movies_df["movieId"].isin(rated_ids)]
            .copy()
        )
        # Map rating values from the user_item_matrix
        rating_map = rated.to_dict()
        history["rating"] = history["movieId"].map(rating_map)
        history = (
            history[["movieId", "title", "year", "genres", "rating"]]
            .sort_values("rating", ascending=False)
            .head(history_n)
        )
        return history, recs
    
    return recs
