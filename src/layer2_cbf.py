"""
Layer 2 – Content-Based Filtering (Personalisation / Re-ranking).

Each product is represented by a TF-IDF vector built from:
    Category + ProductName + Variant + Description   (all lower-cased, Indonesian text)

Given a set of cart product IDs, a "user profile" vector is computed as the mean of
the cart items' TF-IDF vectors. Every product is then scored by cosine similarity to
this profile, giving a personalised relevance score in [0, 1].
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = Path(__file__).parent.parent / "models"


def _make_feature_text(products: pd.DataFrame) -> pd.Series:
    return (
        products["Category"].fillna("").astype(str)
        + " "
        + products["ProductName"].fillna("").astype(str)
        + " "
        + products["Variant"].fillna("").astype(str)
        + " "
        + products["Description"].fillna("").astype(str)
    ).str.lower()


def train_cbf(products: pd.DataFrame) -> tuple[TfidfVectorizer, np.ndarray, list[int]]:
    """Fit TF-IDF on product corpus and persist artefacts."""
    texts = _make_feature_text(products)
    product_ids = products["ProductId"].astype(int).tolist()

    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=1000,
        sublinear_tf=True,
    )
    matrix = tfidf.fit_transform(texts).toarray().astype(np.float32)  # [n_products, vocab]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, MODELS_DIR / "cbf_tfidf.pkl")
    np.save(MODELS_DIR / "cbf_matrix.npy", matrix)
    joblib.dump(product_ids, MODELS_DIR / "cbf_product_ids.pkl")

    print(f"  CBF: {len(product_ids)} products  |  vocab size: {matrix.shape[1]}")
    return tfidf, matrix, product_ids


def load_cbf_model() -> tuple[TfidfVectorizer, np.ndarray, list[int]]:
    tfidf = joblib.load(MODELS_DIR / "cbf_tfidf.pkl")
    matrix = np.load(MODELS_DIR / "cbf_matrix.npy")
    product_ids = joblib.load(MODELS_DIR / "cbf_product_ids.pkl")
    return tfidf, matrix, product_ids


def cbf_scores(
    cart_product_ids: list[int],
    matrix: np.ndarray,
    product_ids: list[int],
) -> np.ndarray:
    """
    Return cosine-similarity CBF scores for every product.

    Output shape: [n_products], values in [0, 1], aligned to `product_ids` order.
    """
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    cart_indices = [pid_to_idx[pid] for pid in cart_product_ids if pid in pid_to_idx]

    if not cart_indices:
        return np.zeros(len(product_ids), dtype=np.float32)

    user_profile = matrix[cart_indices].mean(axis=0, keepdims=True)  # [1, vocab]
    sims = cosine_similarity(user_profile, matrix).flatten().astype(np.float32)

    # Normalise to [0, 1]
    max_sim = sims.max()
    if max_sim > 0:
        sims /= max_sim

    return sims
