"""
Hybrid Recommender – orchestrates all three layers.

Scoring pipeline for a given cart:
  1.  RFM pseudo-profiling   → determines customer segment & weights (a, b)
  2.  Layer 1 (NCF)          → candidate scores  ∈ (0, 1)   [based on co-purchase patterns]
  3.  Layer 2 (CBF)          → re-rank scores    ∈ [0, 1]   [based on content similarity]
  4.  Layer 3 (Bayesian avg) → quality scores    ∈ [0.2, 1] [based on review ratings]

  layer12 = (ncf_score + cbf_score) / 2
  hybrid  = a × layer12  +  b × bayesian_normalised
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import joblib

from .data_preprocessing import load_products, build_product_lookup, resolve_item_id
from .layer1_ncf import load_ncf_model, NeuralCF
from .layer2_cbf import load_cbf_model, cbf_scores
from .layer3_bayesian import load_bayesian_ratings, bayesian_score_normalised
from .rfm_segmentation import load_rfm_models, segment_cart

MODELS_DIR = Path(__file__).parent.parent / "models"


class HybridRecommender:
    """
    Load all trained artefacts once at startup and expose a single
    `recommend(cart_items, top_n, exclude_cart)` method.
    """

    def __init__(self):
        # Product catalogue
        self.products = load_products()
        self.lookup = build_product_lookup(self.products)

        # Ordered list of all product IDs (used to align score arrays)
        self.all_pids: list[int] = self.products["ProductId"].astype(int).tolist()
        self.pid_to_pos: dict[int, int] = {pid: i for i, pid in enumerate(self.all_pids)}

        # Layer 1 – NCF
        self.ncf: NeuralCF = load_ncf_model()
        ncf_cfg = joblib.load(MODELS_DIR / "ncf_config.pkl")
        # ncf_item_ids[i] = product_id for NCF item index i
        self.ncf_item_ids: list[int] = joblib.load(MODELS_DIR / "ncf_item_ids.pkl")
        self.pid_to_ncf_idx: dict[int, int] = {
            pid: i for i, pid in enumerate(self.ncf_item_ids)
        }

        # Layer 2 – CBF
        self.cbf_tfidf, self.cbf_matrix, self.cbf_product_ids = load_cbf_model()
        self.pid_to_cbf_pos: dict[int, int] = {
            pid: i for i, pid in enumerate(self.cbf_product_ids)
        }

        # Layer 3 – Bayesian ratings
        self.ba_dict, self.global_mean = load_bayesian_ratings()

        # RFM segmentation
        self.rfm_scaler, self.rfm_kmeans, self.rfm_label_map = load_rfm_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        cart_items: list[dict[str, Any]],
        top_n: int = 10,
        exclude_cart: bool = True,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        cart_items : list of dicts with keys 'name', 'variant', 'quantity'
        top_n      : number of recommendations to return
        exclude_cart : whether to exclude products already in cart

        Returns
        -------
        dict with customer_segment, weights, cart_summary, and recommendations list
        """
        # ── 0. Resolve cart items ──────────────────────────────────────
        cart_pids, cart_value, identified = self._resolve_cart(cart_items)
        if not cart_pids:
            return {"error": "Tidak ada produk yang dikenali dalam keranjang.", "recommendations": []}

        # ── 1. RFM segmentation ────────────────────────────────────────
        seg = segment_cart(
            cart_value=cart_value,
            n_cart_items=len(cart_pids),
            scaler=self.rfm_scaler,
            kmeans=self.rfm_kmeans,
            label_map=self.rfm_label_map,
        )
        a, b = seg["a"], seg["b"]

        # ── 2. Layer 1 – NCF ───────────────────────────────────────────
        ncf_arr = self._run_ncf(cart_pids)   # aligned to self.all_pids

        # ── 3. Layer 2 – CBF ───────────────────────────────────────────
        cbf_arr = self._run_cbf(cart_pids)   # aligned to self.all_pids

        # ── 4. Layer 1+2 combined ──────────────────────────────────────
        layer12 = (ncf_arr + cbf_arr) / 2.0

        # ── 5. Layer 3 – Bayesian average ─────────────────────────────
        ba_arr = np.array(
            [bayesian_score_normalised(pid, self.ba_dict, self.global_mean) for pid in self.all_pids],
            dtype=np.float32,
        )

        # ── 6. Hybrid score ────────────────────────────────────────────
        hybrid_arr = a * layer12 + b * ba_arr

        # ── 7. Build result list ───────────────────────────────────────
        cart_pid_set = set(cart_pids)
        products_df = self.products.set_index("ProductId")

        rows: list[dict] = []
        for i, pid in enumerate(self.all_pids):
            if exclude_cart and pid in cart_pid_set:
                continue
            try:
                p = products_df.loc[pid]
            except KeyError:
                continue

            rows.append(
                {
                    "product_id": int(pid),
                    "product_name": str(p["ProductName"]),
                    "variant": str(p["Variant"]),
                    "category": str(p["Category"]),
                    "price": float(p["Price"]),
                    "scores": {
                        "layer1_ncf":            round(float(ncf_arr[i]), 4),
                        "layer2_cbf":            round(float(cbf_arr[i]), 4),
                        "layer12_combined":      round(float(layer12[i]), 4),
                        "layer3_bayesian_raw":   round(float(ba_arr[i] * 5), 4),
                        "layer3_bayesian_norm":  round(float(ba_arr[i]), 4),
                        "hybrid_score":          round(float(hybrid_arr[i]), 4),
                    },
                }
            )

        rows.sort(key=lambda x: x["scores"]["hybrid_score"], reverse=True)

        return {
            "customer_segment": seg["segment"],
            "cluster_id": seg["cluster_id"],
            "weights": {"a": a, "b": b},
            "cart_summary": {
                "items_count": len(cart_pids),
                "total_value": round(cart_value, 2),
                "identified_products": identified,
            },
            "recommendations": rows[:top_n],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_cart(
        self, cart_items: list[dict]
    ) -> tuple[list[int], float, list[dict]]:
        cart_pids: list[int] = []
        cart_value = 0.0
        identified: list[dict] = []

        for item in cart_items:
            name = str(item.get("name", "")).strip()
            variant = str(item.get("variant", "-")).strip()
            qty = max(int(item.get("quantity", 1)), 1)

            item_str = f"{name} ({variant})" if variant and variant != "-" else name
            pid = resolve_item_id(item_str, self.lookup, self.products)

            if pid is None:
                continue

            row = self.products[self.products["ProductId"] == pid]
            price = float(row["Price"].iloc[0]) if not row.empty else 0.0
            cart_value += price * qty
            cart_pids.append(pid)
            identified.append(
                {
                    "product_id": int(pid),
                    "name": str(row["ProductName"].iloc[0]) if not row.empty else name,
                    "variant": str(row["Variant"].iloc[0]) if not row.empty else variant,
                    "price": price,
                    "quantity": qty,
                }
            )

        return cart_pids, cart_value, identified

    def _run_ncf(self, cart_pids: list[int]) -> np.ndarray:
        """Run NCF and return scores aligned to self.all_pids."""
        cart_ncf_indices = [
            self.pid_to_ncf_idx[pid] for pid in cart_pids if pid in self.pid_to_ncf_idx
        ]
        if not cart_ncf_indices:
            return np.zeros(len(self.all_pids), dtype=np.float32)

        # raw scores aligned to ncf_item_ids order
        raw_scores = self.ncf.score_with_virtual_user(cart_ncf_indices)

        # Normalise to [0, 1]
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            raw_scores = (raw_scores - s_min) / (s_max - s_min)

        # Re-align to self.all_pids
        ncf_by_pid = {self.ncf_item_ids[i]: raw_scores[i] for i in range(len(self.ncf_item_ids))}
        return np.array(
            [ncf_by_pid.get(pid, 0.0) for pid in self.all_pids], dtype=np.float32
        )

    def _run_cbf(self, cart_pids: list[int]) -> np.ndarray:
        """Run CBF and return scores aligned to self.all_pids."""
        # cbf_scores returns array aligned to self.cbf_product_ids
        cbf_arr = cbf_scores(cart_pids, self.cbf_matrix, self.cbf_product_ids)
        cbf_by_pid = {self.cbf_product_ids[i]: cbf_arr[i] for i in range(len(self.cbf_product_ids))}
        return np.array(
            [cbf_by_pid.get(pid, 0.0) for pid in self.all_pids], dtype=np.float32
        )
