import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Load the processed book dataset
df = pd.read_csv("books_with_clusters.csv")

# Load the TF-IDF matrix created by Part A
X = load_npz("tfidf_matrix.npz")

# Compute cosine similarity between all books
cosine_sim = cosine_similarity(X, X)

# Create a mapping from book title to row index
indices = pd.Series(df.index, index=df["Title"]).drop_duplicates()


def recommend_books(title, top_n=10):
    """
    Recommend top_n books similar to the given title
    based on cosine similarity of TF-IDF features.
    """

    # Check whether the title exists
    if title not in indices:
        return f"Book '{title}' not found in the dataset."

    # Get the index of the selected book
    idx = indices[title]

    # Get pairwise similarity scores for that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort books by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the book itself and keep top_n similar books
    sim_scores = sim_scores[1:top_n + 1]

    # Get indices of recommended books
    book_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    # Build result dataframe
    recommendations = df.iloc[book_indices][["Title", "Authors", "keyword_category", "Description"]].copy()
    recommendations["similarity_score"] = similarity_scores

    return recommendations
