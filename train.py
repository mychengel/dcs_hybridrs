"""
Training pipeline for the Hybrid Recommendation System.

Run once before starting the API:
    python train.py

All model artefacts are saved to the  models/  directory.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow imports from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import (
    build_product_lookup,
    expand_transactions,
    load_products,
    load_reviews,
    load_transactions,
    resolve_item_id,
)
from src.layer1_ncf import train_ncf
from src.layer2_cbf import train_cbf
from src.layer3_bayesian import compute_bayesian_ratings
from src.rfm_segmentation import compute_transaction_rfm, train_rfm

MODELS_DIR = Path("models")


def _banner(text: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main():
    t0 = time.time()
    _banner("Hybrid Recommendation System — Training")

    # ── 1. Load raw data ────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    products = load_products()
    reviews = load_reviews()
    transactions = load_transactions()
    print(f"  Products    : {len(products):,}")
    print(f"  Reviews     : {len(reviews):,}")
    print(f"  Transactions: {len(transactions):,}")

    # ── 2. Parse transaction items → session-item pairs ─────────────────
    print("\n[2/6] Parsing transaction items …")
    lookup = build_product_lookup(products)

    # Sorted product IDs → contiguous NCF item indices
    all_pids: list[int] = sorted(products["ProductId"].astype(int).unique())
    pid_to_ncf_idx: dict[int, int] = {pid: i for i, pid in enumerate(all_pids)}
    n_items = len(all_pids)

    session_item_records = []
    for sess_idx, row in transactions.iterrows():
        raw_items = str(row["Items"])
        item_strings = [s.strip() for s in raw_items.split(",") if s.strip()]
        for item_str in item_strings:
            pid = resolve_item_id(item_str, lookup, products)
            if pid is not None and pid in pid_to_ncf_idx:
                session_item_records.append(
                    {
                        "session_idx": sess_idx,          # temporary — may not be contiguous
                        "item_idx": pid_to_ncf_idx[pid],
                        "product_id": pid,
                    }
                )

    session_item_df = pd.DataFrame(session_item_records)
    print(f"  Identified session-item pairs: {len(session_item_df):,}")

    # Remap session indices to be contiguous (required by Embedding layer)
    unique_sessions = session_item_df["session_idx"].unique()
    sess_to_idx = {s: i for i, s in enumerate(unique_sessions)}
    session_item_df["session_idx"] = session_item_df["session_idx"].map(sess_to_idx)
    n_sessions = len(unique_sessions)
    print(f"  Unique sessions (transactions with known items): {n_sessions:,}")
    print(f"  Unique NCF item indices: {n_items}")

    # Persist item-ID mapping so the recommender can align scores later
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_pids, MODELS_DIR / "ncf_item_ids.pkl")
    joblib.dump(pid_to_ncf_idx, MODELS_DIR / "ncf_pid_to_idx.pkl")

    # ── 3. RFM segmentation ─────────────────────────────────────────────
    print("\n[3/6] Training RFM segmentation …")
    rfm = compute_transaction_rfm(transactions)
    scaler, kmeans, label_map = train_rfm(rfm)
    print(f"  Cluster → segment map: {label_map}")

    # ── 4. Layer 1 – NCF ────────────────────────────────────────────────
    print("\n[4/6] Training NCF (Layer 1) …")
    train_ncf(
        session_item_df=session_item_df,
        n_sessions=n_sessions,
        n_items=n_items,
        epochs=15,
        batch_size=2048,
        lr=1e-3,
        neg_ratio=4,
    )

    # ── 5. Layer 2 – CBF ────────────────────────────────────────────────
    print("\n[5/6] Training CBF (Layer 2) …")
    train_cbf(products)

    # ── 6. Layer 3 – Bayesian ratings ───────────────────────────────────
    print("\n[6/6] Computing Bayesian ratings (Layer 3) …")
    compute_bayesian_ratings(reviews, products)

    elapsed = time.time() - t0
    _banner(f"Training complete in {elapsed:.1f}s — artefacts saved to models/")
    print(
        "\nStart the API with:\n"
        "  uvicorn api:app --host 0.0.0.0 --port 8000 --reload\n"
    )


if __name__ == "__main__":
    main()
