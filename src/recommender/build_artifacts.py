import os
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data")
ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def main():
    # 1. Load raw data
    movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

    # 2. Preprocess movies (extract year, clean title)
    movies_df = movies_df.copy()
    movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)\s*$").astype("Int64")
    movies_df["title"] = movies_df["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    )

    # 3. Build user–item matrix
    user_item_matrix = ratings_df.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    # 4. Build item–item similarity matrix
    item_user_matrix = user_item_matrix.T
    item_similarity = cosine_similarity(item_user_matrix)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=item_user_matrix.index,
        columns=item_user_matrix.index,
    )

    # 5. Save artifacts
    joblib.dump(movies_df, os.path.join(ARTIFACT_DIR, "movies_df.pkl"))
    joblib.dump(ratings_df, os.path.join(ARTIFACT_DIR, "ratings_df.pkl"))
    joblib.dump(user_item_matrix, os.path.join(ARTIFACT_DIR, "user_item_matrix.pkl"))
    joblib.dump(item_similarity_df, os.path.join(ARTIFACT_DIR, "item_similarity_df.pkl"))

    print("Saved artifacts to", ARTIFACT_DIR)

if __name__ == "__main__":
    main()

