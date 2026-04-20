"""
RFM Pseudo-Profiling & Customer Segmentation.

Each anonymous transaction is treated as a data point with:
  R = days since transaction (from dataset end date) — higher means older/more at-risk
  F = TotalItem (number of items bought)
  M = TotalAmount (total spend)

KMeans (N=4) is trained once and the cluster centroids are mapped to semantic labels:
  Cluster 2 → Loyal Customer     (a=0.5, b=0.5)
  Cluster 1 → New Customer       (a=0.6, b=0.4)
  Cluster 3 → At Risk Customer   (a=0.7, b=0.3)
  Cluster 0 → Lost Customer      (a=0.8, b=0.2)

At inference the cart's (F, M) is used as a pseudo-RFM profile (R=0, just purchasing)
and assigned to the nearest cluster centroid.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).parent.parent / "models"

# Weights keyed by semantic label
SEGMENT_WEIGHTS: dict[str, dict] = {
    "Loyal Customer":    {"a": 0.5, "b": 0.5},
    "New Customer":      {"a": 0.6, "b": 0.4},
    "At Risk Customer":  {"a": 0.7, "b": 0.3},
    "Lost Customer":     {"a": 0.8, "b": 0.2},
}


def compute_transaction_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-transaction RFM features."""
    end_date = transactions["Date"].max()
    rfm = pd.DataFrame(
        {
            "R": (end_date - transactions["Date"]).dt.days.astype(float),
            "F": transactions["TotalItem"].astype(float),
            "M": transactions["TotalAmount"].astype(float),
        }
    ).dropna()
    return rfm


def _assign_labels(kmeans: KMeans, scaler: StandardScaler) -> dict[int, str]:
    """
    Map KMeans cluster IDs to semantic labels based on centroid characteristics.

    Strategy (greedy, in order of most-distinguishable feature):
      1. Lost Customer    → highest R centroid overall
      2. Loyal Customer   → highest F+M among the remaining three
      3. At Risk Customer → highest R among the remaining two
      4. New Customer     → the leftover cluster
    """
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    df = pd.DataFrame(centroids, columns=["R", "F", "M"])
    remaining = list(range(len(df)))
    label_map: dict[int, str] = {}

    # 1. Lost Customer: paling tinggi R
    lost = int(df.loc[remaining, "R"].idxmax())
    label_map[lost] = "Lost Customer"
    remaining = [i for i in remaining if i != lost]

    # 2. Loyal Customer: tinggi F dan M
    loyal = int((df.loc[remaining, "F"] + df.loc[remaining, "M"]).idxmax())
    label_map[loyal] = "Loyal Customer"
    remaining = [i for i in remaining if i != loyal]

    # 3. At Risk Customer: tinggi R di antara sisa
    at_risk = int(df.loc[remaining, "R"].idxmax())
    label_map[at_risk] = "At Risk Customer"
    remaining = [i for i in remaining if i != at_risk]

    # 4. New Customer: sisa terakhir
    label_map[remaining[0]] = "New Customer"

    return label_map


def train_rfm(rfm: pd.DataFrame) -> tuple[StandardScaler, KMeans, dict[int, str]]:
    """Train and persist the RFM scaler + KMeans model."""
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["R", "F", "M"]])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X)

    label_map = _assign_labels(kmeans, scaler)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "rfm_scaler.pkl")
    joblib.dump(kmeans, MODELS_DIR / "rfm_kmeans.pkl")
    joblib.dump(label_map, MODELS_DIR / "rfm_label_map.pkl")

    return scaler, kmeans, label_map


def load_rfm_models() -> tuple[StandardScaler, KMeans, dict[int, str]]:
    scaler = joblib.load(MODELS_DIR / "rfm_scaler.pkl")
    kmeans = joblib.load(MODELS_DIR / "rfm_kmeans.pkl")
    label_map = joblib.load(MODELS_DIR / "rfm_label_map.pkl")
    return scaler, kmeans, label_map


def segment_cart(
    cart_value: float,
    n_cart_items: int,
    scaler: StandardScaler,
    kmeans: KMeans,
    label_map: dict[int, str],
) -> dict:
    """
    Determine customer segment from current cart using pseudo-RFM.
    R=0 (customer is purchasing right now), F=n_cart_items, M=cart_value.
    """
    rfm_vec = np.array([[0.0, float(n_cart_items), float(cart_value)]])
    scaled = scaler.transform(rfm_vec)
    cluster_id = int(kmeans.predict(scaled)[0])
    segment = label_map.get(cluster_id, "New Customer")
    weights = SEGMENT_WEIGHTS.get(segment, {"a": 0.5, "b": 0.5})

    return {
        "cluster_id": cluster_id,
        "segment": segment,
        "a": weights["a"],
        "b": weights["b"],
    }
