# B_new.py — Member B
# Classification models (NB, LR, RF) + Cosine Similarity Recommender
#
# When imported as a module: only the cosine recommender is loaded.
# When run directly (python B_new.py): full pipeline including classification,
# evaluation, confusion matrices, and visualizations.

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Module-level: data + cosine recommender (available on import)
# ============================================================

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


# ============================================================
# Main: classification pipeline + evaluation + visualizations
# (only runs when executed directly: python B_new.py)
# ============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Step 0 — Load data & features
    # ----------------------------------------------------------
    import joblib

    print(df.columns)

    # Use keyword_category as label
    y = df["keyword_category"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(y.value_counts().head())

    # ----------------------------------------------------------
    # Step 1 — Train/Test Split (80/20)
    # ----------------------------------------------------------
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # ----------------------------------------------------------
    # Step 2 — Train Three Classification Models
    # ----------------------------------------------------------

    # 2.1 Naive Bayes
    from sklearn.naive_bayes import MultinomialNB

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    print("Naive Bayes training completed.")

    # 2.2 Logistic Regression
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    print("Logistic Regression training completed.")

    # 2.3 Random Forest
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print("Random Forest training completed.")

    # ----------------------------------------------------------
    # Step 3 — Model Evaluation (Precision / Recall / F1-score)
    # ----------------------------------------------------------
    from sklearn.metrics import classification_report, accuracy_score

    print("===== Naive Bayes =====")
    print(classification_report(y_test, nb_pred))
    print("Accuracy:", accuracy_score(y_test, nb_pred))

    print("\n===== Logistic Regression =====")
    print(classification_report(y_test, lr_pred))
    print("Accuracy:", accuracy_score(y_test, lr_pred))

    print("\n===== Random Forest =====")
    print(classification_report(y_test, rf_pred))
    print("Accuracy:", accuracy_score(y_test, rf_pred))

    # Summary Table
    from sklearn.metrics import precision_recall_fscore_support

    results = []
    for name, pred in {
        "Naive Bayes": nb_pred,
        "Logistic Regression": lr_pred,
        "Random Forest": rf_pred
    }.items():
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0
        )
        acc = accuracy_score(y_test, pred)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

    results_df = pd.DataFrame(results)
    print(results_df)

    # ----------------------------------------------------------
    # Step 4 — Confusion Matrix
    # ----------------------------------------------------------
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    labels = sorted(y.unique())

    # Naive Bayes
    cm_nb = confusion_matrix(y_test, nb_pred, labels=labels)
    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=labels)
    disp_nb.plot(xticks_rotation=45)
    plt.title("Confusion Matrix - Naive Bayes")
    plt.show()

    # Logistic Regression
    cm_lr = confusion_matrix(y_test, lr_pred, labels=labels)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=labels)
    disp_lr.plot(xticks_rotation=45)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    # Random Forest
    cm_rf = confusion_matrix(y_test, rf_pred, labels=labels)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=labels)
    disp_rf.plot(xticks_rotation=45)
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    # ----------------------------------------------------------
    # Step 5 — Error Analysis
    # ----------------------------------------------------------

    # 5.1 Attach predictions to test dataset
    df_test = df_test.copy()
    df_test["true_label"] = y_test.values
    df_test["NB_pred"] = nb_pred
    df_test["LR_pred"] = lr_pred
    df_test["RF_pred"] = rf_pred

    # 5.2 Logistic Regression errors
    lr_errors = df_test[df_test["true_label"] != df_test["LR_pred"]]
    print("Number of LR misclassified samples:", len(lr_errors))
    print(lr_errors[["Title", "true_label", "LR_pred", "Description"]].head(10))

    # 5.3 Compare model errors
    nb_error_count = (df_test["true_label"] != df_test["NB_pred"]).sum()
    lr_error_count = (df_test["true_label"] != df_test["LR_pred"]).sum()
    rf_error_count = (df_test["true_label"] != df_test["RF_pred"]).sum()

    error_compare = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression", "Random Forest"],
        "Number of Errors": [nb_error_count, lr_error_count, rf_error_count]
    })
    error_compare = error_compare.sort_values(by="Number of Errors")
    print(error_compare)

    # 5.4 Visualization of errors
    plt.figure(figsize=(8, 5))
    plt.bar(error_compare["Model"], error_compare["Number of Errors"])
    plt.title("Number of Misclassified Samples by Model")
    plt.xlabel("Model")
    plt.ylabel("Number of Errors")
    plt.show()

    # 5.5 Most confused categories
    lr_confusions = (
        lr_errors.groupby(["true_label", "LR_pred"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )
    print(lr_confusions.head(10))

    # ----------------------------------------------------------
    # Cosine Similarity Recommender — Tests & Visualization
    # ----------------------------------------------------------

    print("\nDataset shape:", df.shape)
    print("TF-IDF matrix shape:", X.shape)
    print(df[["Title", "keyword_category"]].head())
    print("Cosine similarity matrix shape:", cosine_sim.shape)
    print("Number of unique titles:", len(indices))

    # Step 5 — Test the recommender
    print(df["Title"].head(20).tolist())
    print(recommend_books("The Great Gatsby", top_n=10))

    # Step 6 — Show one sample recommendation nicely
    sample_title = df["Title"].iloc[0]
    print("Selected book:", sample_title)
    print(recommend_books(sample_title, top_n=10))

    # Step 7 — Add a simple visualization
    sample_title = df["Title"].iloc[0]
    result = recommend_books(sample_title, top_n=10)

    plt.figure(figsize=(10, 5))
    plt.bar(result["Title"], result["similarity_score"])
    plt.xticks(rotation=75)
    plt.title(f"Top 10 Recommendations for '{sample_title}'")
    plt.xlabel("Recommended Books")
    plt.ylabel("Cosine Similarity Score")
    plt.tight_layout()
    plt.show()
