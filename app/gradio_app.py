import joblib
import gradio as gr
import pandas as pd

from recommender.item_item_cf import recommend_similar, recommend_for_user

# Load artifacts
movies_df = joblib.load("artifacts/movies_df.pkl")
user_item_matrix = joblib.load("artifacts/user_item_matrix.pkl")
item_similarity_df = joblib.load("artifacts/item_similarity_df.pkl")


def similar_movies(movie_title: str, n: int = 5):
    recs = recommend_similar(
        movie=movie_title,
        movies_df=movies_df,
        item_similarity_df=item_similarity_df,
        n=n,
    )

    # Debug: see what columns we actually have
    print("similar_movies recs.columns:", list(recs.columns))

    # Try to infer reasonable columns: id, title, score-like
    possible_score_cols = ["score", "similarity", "sim_score"]
    score_col = next((c for c in possible_score_cols if c in recs.columns), None)

    possible_id_cols = ["movie_id", "movieId", "item_id", "itemId"]
    id_col = next((c for c in possible_id_cols if c in recs.columns), None)

    cols = ["title"]
    if id_col:
        cols.insert(0, id_col)
    if score_col:
        cols.append(score_col)

    # Only keep columns that actually exist
    cols = [c for c in cols if c in recs.columns]

    # Fallback: if something went weird, just return the whole df
    if not cols:
        return recs

    return recs[cols]


def recommend_for_user_ui(user_id: int, n: int = 5):
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

    # Debug: see what columns we actually have
    print("recommend_for_user recs.columns:", list(recs.columns))

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


with gr.Blocks(title="🎬 Movie Recommender") as demo:
    gr.Markdown("# 🎬 Movie Recommender (Item–Item CF)")

    with gr.Tab("Similar Movies"):
        movie_in = gr.Textbox(label="Movie title", value="Toy Story")
        n_sim = gr.Slider(1, 20, value=5, step=1, label="Number of results")
        sim_btn = gr.Button("Get similar movies")
        sim_out = gr.Dataframe(label="Similar movies")
        sim_btn.click(similar_movies, inputs=[movie_in, n_sim], outputs=sim_out)

    with gr.Tab("User Recommendations"):
        user_in = gr.Number(label="User ID", value=1, precision=0)
        n_rec = gr.Slider(1, 20, value=5, step=1, label="Number of results")
        rec_btn = gr.Button("Get recommendations")
        rec_out = gr.Dataframe(label="Recommendations")
        rec_btn.click(recommend_for_user_ui, inputs=[user_in, n_rec], outputs=rec_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
