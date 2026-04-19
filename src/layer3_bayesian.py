"""
Layer 3 – Bayesian Average Rating (Quality / Social Proof).

For each product the Bayesian Average is:

    BA = (C × m + Σ ratings) / (C + n_ratings)

where
    m = global mean rating across all reviews
    C = mean number of ratings per product (prior strength)

Products with no reviews are assigned the global mean m.

The raw score (1–5 scale) is normalised to [0.2, 1.0] by dividing by 5.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"


def compute_bayesian_ratings(
    reviews: pd.DataFrame, products: pd.DataFrame
) -> dict[int, float]:
    """
    Compute and persist Bayesian average ratings for all products.
    Returns {product_id: bayesian_avg_rating}.
    """
    global_mean = float(reviews["Rating"].mean())

    per_product = (
        reviews.groupby("Menu")["Rating"]
        .agg(n_ratings="count", sum_ratings="sum")
        .reset_index()
        .rename(columns={"Menu": "ProductId"})
    )
    C = float(per_product["n_ratings"].mean())

    per_product["bayesian_avg"] = (
        C * global_mean + per_product["sum_ratings"]
    ) / (C + per_product["n_ratings"])

    ba_dict: dict[int, float] = {
        int(row["ProductId"]): float(row["bayesian_avg"])
        for _, row in per_product.iterrows()
    }

    # Products with zero reviews → global mean
    for pid in products["ProductId"].astype(int).tolist():
        if pid not in ba_dict:
            ba_dict[pid] = global_mean

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"ba_dict": ba_dict, "global_mean": global_mean, "C": C},
        MODELS_DIR / "bayesian_ratings.pkl",
    )

    reviewed = len(per_product)
    total = len(products)
    print(
        f"  Bayesian ratings: {reviewed}/{total} products reviewed  "
        f"|  global_mean={global_mean:.3f}  C={C:.1f}"
    )
    return ba_dict


def load_bayesian_ratings() -> tuple[dict[int, float], float]:
    """Returns (ba_dict, global_mean)."""
    data = joblib.load(MODELS_DIR / "bayesian_ratings.pkl")
    return data["ba_dict"], data["global_mean"]


def bayesian_score_normalised(product_id: int, ba_dict: dict[int, float], global_mean: float = 4.0) -> float:
    """Return Bayesian average normalised to [0.2, 1.0] by dividing by 5."""
    raw = ba_dict.get(int(product_id), global_mean)
    return float(raw) / 5.0
