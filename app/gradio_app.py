import joblib
import gradio as gr
import pandas as pd

from recommender.item_item_cf import recommend_similar, recommend_for_user
from recommender.funk_mf import FunkMF  

# ------------------ Load artifacts ------------------

movies_df = joblib.load("artifacts/movies_df.pkl")
user_item_matrix = joblib.load("artifacts/user_item_matrix.pkl")
item_similarity_df = joblib.load("artifacts/item_similarity_df.pkl")

# ratings_df should be saved with columns like: userId, movieId, rating
ratings_df = joblib.load("artifacts/ratings_df.pkl")

# ------------------ Load FunkMF model ------------------

mf_model = joblib.load("artifacts/funk_mf_model.pkl")

# ------------------ Item–Item CF helper functions ------------------

def similar_movies(movie_title: str, n: int = 5):
    """Wrapper for item–item CF similar-movie search."""
    recs = recommend_similar(
        movie=movie_title,
        movies_df=movies_df,
        item_similarity_df=item_similarity_df,
        n=n,
    )

    # Try to infer id and score columns
    possible_score_cols = ["score", "similarity", "sim_score"]
    score_col = next((c for c in possible_score_cols if c in recs.columns), None)

    possible_id_cols = ["movie_id", "movieId", "item_id", "itemId"]
    id_col = next((c for c in possible_id_cols if c in recs.columns), None)

    cols = ["title"]
    if id_col:
        cols.insert(0, id_col)
    if score_col:
        cols.append(score_col)

    # Only keep columns that exist
    cols = [c for c in cols if c in recs.columns]

    # Fallback: if nothing matches, return the full DataFrame
    if not cols:
        return recs

    return recs[cols]


def recommend_for_user_ui(user_id: int, n: int = 5):
    """Wrapper for item–item CF user-based recommendations."""
    recs = recommend_for_user(
        user_id=int(user_id),
        movies_df=movies_df,
        user_item_matrix=user_item_matrix,
        item_similarity_df=item_similarity_df,
        n_recs=n,
    )

    if recs is None or recs.empty:
        return pd.DataFrame(
            [{"movie_id": None, "title": "No recommendations", "score": None}]
        )

    possible_score_cols = ["score", "similarity", "sim_score"]
    score_col = next((c for c in possible_score_cols if c in recs.columns), None)

    possible_id_cols = ["movie_id", "movieId", "item_id", "itemId"]
    id_col = next((c for c in possible_id_cols if c in recs.columns), None)

    cols = ["title"]
    if id_col:
        cols.insert(0, id_col)
    if score_col:
        cols.append(score_col)

    cols = [c for c in cols if c in recs.columns]

    if not cols:
        return recs

    return recs[cols]


# ------------------ FunkMF helper function ------------------

def mf_recommend_for_user_ui(user_id: int, n: int = 5):
    """Use FunkMF to recommend top-N movies for a user."""
    recs = mf_model.recommend_top_n(
        user_id=int(user_id),
        movies_df=movies_df,
        ratings_df=ratings_df,   # filters out movies the user has already rated
        n=n,
        min_rating=0.5,
        max_rating=5.0,
    )

    # recs is a list of (title, movieId, predicted_rating)
    rows = [
        {"movieId": mid, "title": title, "pred_rating": score}
        for title, mid, score in recs
    ]
    return pd.DataFrame(rows)


# ------------------ Gradio UI ------------------

with gr.Blocks(title="🎬 Movie Recommender") as demo:
    gr.Markdown("# 🎬 Movie Recommender (Item–Item CF + FunkMF)")

    with gr.Tab("Similar Movies"):
        movie_in = gr.Textbox(label="Movie title", value="Toy Story")
        n_sim = gr.Slider(1, 20, value=5, step=1, label="Number of results")
        sim_btn = gr.Button("Get similar movies")
        sim_out = gr.Dataframe(label="Similar movies")
        sim_btn.click(similar_movies, inputs=[movie_in, n_sim], outputs=sim_out)

    with gr.Tab("User Recommendations (Item–Item CF)"):
        user_in = gr.Number(label="User ID", value=1, precision=0)
        n_rec = gr.Slider(1, 20, value=5, step=1, label="Number of results")
        rec_btn = gr.Button("Get recommendations")
        rec_out = gr.Dataframe(label="Recommendations")
        rec_btn.click(recommend_for_user_ui, inputs=[user_in, n_rec], outputs=rec_out)

    with gr.Tab("User Recommendations (FunkMF)"):
        mf_user_in = gr.Number(label="User ID", value=1, precision=0)
        mf_n_rec = gr.Slider(1, 20, value=5, step=1, label="Number of results")
        mf_rec_btn = gr.Button("Get MF recommendations")
        mf_rec_out = gr.Dataframe(label="MF Recommendations")
        mf_rec_btn.click(
            mf_recommend_for_user_ui,
            inputs=[mf_user_in, mf_n_rec],
            outputs=mf_rec_out,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
