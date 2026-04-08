# -*- coding: utf-8 -*-
"""
B_new.py

Behavior:
- When imported as a module: only the cosine recommender is loaded.
- When run directly (python B_new.py): full pipeline including classification,
  evaluation, confusion matrices, and visualizations.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# RECOMMENDER (MODULE-LEVEL)
# =========================================================

def load_recommender_assets(data_path="books_with_clusters.csv", tfidf_path="tfidf_matrix.npz"):
    """Load dataset and TF-IDF matrix for the recommender."""
    df = pd.read_csv(data_path)
    tfidf_matrix = load_npz(tfidf_path)
    return df, tfidf_matrix


def build_title_index(df, title_col="Title"):
    """Build a title-to-index mapping."""
    return pd.Series(df.index, index=df[title_col]).drop_duplicates()


def build_cosine_similarity_matrix(tfidf_matrix):
    """Compute cosine similarity matrix from TF-IDF features."""
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_books(title, df, indices, cosine_sim, top_n=10):
    """Recommend top_n similar books for a given title."""
    if title not in indices:
        return f"Book '{title}' not found in the dataset."

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    book_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    recommendations = df.iloc[book_indices][[
        "Title", "Authors", "keyword_category", "Description"
    ]].copy()
    recommendations["similarity_score"] = similarity_scores

    return recommendations


def test_recommender(data_path="books_with_clusters.csv", tfidf_path="tfidf_matrix.npz", sample_title=None, top_n=10):
    """Run a simple recommender test and visualization."""
    df, tfidf_matrix = load_recommender_assets(data_path, tfidf_path)
    indices = build_title_index(df)
    cosine_sim = build_cosine_similarity_matrix(tfidf_matrix)

    if sample_title is None:
        sample_title = df["Title"].iloc[0]

    print("\n========== Recommender Test ==========")
    print("Dataset shape:", df.shape)
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Selected book:", sample_title)

    result = recommend_books(sample_title, df, indices, cosine_sim, top_n=top_n)
    print(result)

    if isinstance(result, pd.DataFrame):
        plt.figure(figsize=(10, 5))
        plt.bar(result["Title"], result["similarity_score"])
        plt.xticks(rotation=75)
        plt.title(f"Top {top_n} Recommendations for '{sample_title}'")
        plt.xlabel("Recommended Books")
        plt.ylabel("Cosine Similarity Score")
        plt.tight_layout()
        plt.show()

    return result


# =========================================================
# CLASSIFICATION (DIRECT-RUN ONLY)
# =========================================================

def run_classification(data_path="books_with_clusters.csv"):
    """
    Full classification pipeline.
    This is intentionally defined to import classification dependencies only
    when the script is run directly.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from gensim.models import Word2Vec

    def document_vector(tokens, model, dim):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(dim)
        return np.mean(vectors, axis=0)

    def load_and_prepare_data(path):
        df_local = pd.read_csv(path)
        df_local = df_local.dropna(subset=["text_clean", "keyword_category"]).copy()
        df_local["text_clean"] = df_local["text_clean"].astype(str)
        df_local["keyword_category"] = df_local["keyword_category"].astype(str)
        return df_local

    def build_feature_sets(X_train_text, X_test_text):
        bow_vectorizer = CountVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        X_train_bow = bow_vectorizer.fit_transform(X_train_text)
        X_test_bow = bow_vectorizer.transform(X_test_text)

        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

        lda_model = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            learning_method="batch"
        )
        X_train_lda = lda_model.fit_transform(X_train_bow)
        X_test_lda = lda_model.transform(X_test_bow)

        train_tokens = [text.split() for text in X_train_text]
        test_tokens = [text.split() for text in X_test_text]

        w2v_model = Word2Vec(
            sentences=train_tokens,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            seed=42
        )

        embedding_dim = w2v_model.vector_size
        X_train_embed = np.array([
            document_vector(tokens, w2v_model, embedding_dim)
            for tokens in train_tokens
        ])
        X_test_embed = np.array([
            document_vector(tokens, w2v_model, embedding_dim)
            for tokens in test_tokens
        ])

        return {
            "BoW": (X_train_bow, X_test_bow),
            "TF-IDF": (X_train_tfidf, X_test_tfidf),
            "LDA": (X_train_lda, X_test_lda),
            "WordEmbedding": (X_train_embed, X_test_embed)
        }

    def get_top_misclassified_full(feature, model_name, df_test, y_test_series, all_predictions, top_n=5):
        y_pred = all_predictions[(feature, model_name)]["y_pred"]
        temp_df = df_test.copy()
        temp_df["true_label"] = y_test_series.values
        temp_df["predicted_label"] = y_pred

        errors_df = temp_df[temp_df["true_label"] != temp_df["predicted_label"]].copy()
        errors_df["text_length"] = errors_df["text_clean"].apply(len)
        errors_df = errors_df.sort_values(by="text_length", ascending=False)
        return errors_df.head(top_n)

    df = load_and_prepare_data(data_path)

    print("Dataset shape:", df.shape)
    print(df[["Title", "text_clean", "keyword_category"]].head())
    print("\nCategory distribution:")
    print(df["keyword_category"].value_counts())

    X_text = df["text_clean"]
    y = df["keyword_category"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", len(X_train_text))
    print("Test size:", len(X_test_text))

    feature_sets = build_feature_sets(X_train_text, X_test_text)
    for name, (Xtr, Xte) in feature_sets.items():
        print(f"{name} -> train: {Xtr.shape}, test: {Xte.shape}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM": LinearSVC(random_state=42)
    }

    results = []
    all_predictions = {}

    for feature_name, (Xtr, Xte) in feature_sets.items():
        print(f"\n========== Feature: {feature_name} ==========")
        for model_name, model in models.items():
            print(f"Training {model_name} on {feature_name}...")
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)

            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )

            results.append({
                "Feature": feature_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1
            })

            all_predictions[(feature_name, model_name)] = {
                "model": model,
                "y_pred": y_pred
            }

            print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["F1-score", "Accuracy"], ascending=False).reset_index(drop=True)

    print("\n========== Summary Table ==========")
    print(results_df)

    plt.figure(figsize=(12, 6))
    for model_name in results_df["Model"].unique():
        subset = results_df[results_df["Model"] == model_name]
        plt.plot(subset["Feature"], subset["F1-score"], marker="o", label=model_name)

    plt.title("F1-score Comparison Across Features and Models")
    plt.xlabel("Feature Representation")
    plt.ylabel("Weighted F1-score")
    plt.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    best_row = results_df.iloc[0]
    second_best_row = results_df.iloc[1]

    best_feature = best_row["Feature"]
    best_model_name = best_row["Model"]
    second_best_feature = second_best_row["Feature"]
    second_best_model_name = second_best_row["Model"]

    best_pred = all_predictions[(best_feature, best_model_name)]["y_pred"]
    second_best_pred = all_predictions[(second_best_feature, second_best_model_name)]["y_pred"]

    print(f"\nBest combination: {best_model_name} + {best_feature}\n")
    print(classification_report(y_test, best_pred, zero_division=0))

    print(f"\nSecond-best combination: {second_best_model_name} + {second_best_feature}\n")
    print(classification_report(y_test, second_best_pred, zero_division=0))

    labels = sorted(y.unique())

    cm = confusion_matrix(y_test, best_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix: {best_model_name} + {best_feature}")
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(y_test, second_best_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix: {second_best_model_name} + {second_best_feature}")
    plt.tight_layout()
    plt.show()

    df_test_full = df.loc[X_test_text.index].copy()
    top1_errors = get_top_misclassified_full(best_feature, best_model_name, df_test_full, y_test, all_predictions)
    top2_errors = get_top_misclassified_full(second_best_feature, second_best_model_name, df_test_full, y_test, all_predictions)

    print(f"\nTop 5 misclassified samples for {best_model_name} + {best_feature}:")
    print(top1_errors[["Title", "true_label", "predicted_label", "Description"]])

    print(f"\nTop 5 misclassified samples for {second_best_model_name} + {second_best_feature}:")
    print(top2_errors[["Title", "true_label", "predicted_label", "Description"]])

    return {
        "results_df": results_df,
        "best_row": best_row,
        "second_best_row": second_best_row,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    """Run the full pipeline only when the script is executed directly."""
    run_classification(data_path="books_with_clusters.csv")
    test_recommender(
        data_path="books_with_clusters.csv",
        tfidf_path="tfidf_matrix.npz",
        sample_title=None,
        top_n=10,
    )


if __name__ == "__main__":
    main()
