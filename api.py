"""
FastAPI service – Hybrid Recommendation System.

Endpoints
---------
GET  /health             liveness probe
GET  /products           list the full product catalogue
POST /recommend          main recommendation endpoint (accepts cart, returns top-N)

Run
---
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from src.hybrid_recommender import HybridRecommender


# ---------------------------------------------------------------------------
# Lifespan — load models once at startup
# ---------------------------------------------------------------------------

_recommender: Optional[HybridRecommender] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _recommender
    print("Loading Hybrid Recommender …")
    _recommender = HybridRecommender()
    print("Ready.")
    yield
    _recommender = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hybrid Recommendation System API",
    description=(
        "Multi-layer hybrid recommender for a coffee-shop menu:\n\n"
        "- **Layer 1** – Neural Collaborative Filtering (co-purchase patterns)\n"
        "- **Layer 2** – Content-Based Filtering (product text embeddings)\n"
        "- **Layer 3** – Bayesian Average rating (social proof)\n\n"
        "Customer segment (RFM pseudo-profiling) determines the blending weights."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CartItem(BaseModel):
    name: str = Field(..., examples=["Americano"], description="Nama produk")
    variant: str = Field(
        default="-",
        examples=["Hot Houseblend"],
        description="Varian produk, kosongkan atau '-' jika tidak ada",
    )
    quantity: int = Field(default=1, ge=1, description="Jumlah item dalam keranjang")

    model_config = {"json_schema_extra": {"example": {"name": "Americano", "variant": "Hot Houseblend", "quantity": 1}}}


class RecommendRequest(BaseModel):
    cart: list[CartItem] = Field(
        ...,
        min_length=1,
        description="Daftar item yang sedang ada di keranjang belanja",
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Jumlah rekomendasi yang dikembalikan",
    )
    exclude_cart: bool = Field(
        default=True,
        description="Jika True, produk yang sudah ada di keranjang tidak akan direkomendasikan",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "cart": [
                    {"name": "Americano", "variant": "Hot Houseblend", "quantity": 1},
                    {"name": "Croffle", "variant": "-", "quantity": 1},
                ],
                "top_n": 10,
                "exclude_cart": True,
            }
        }
    }


class ScoreDetail(BaseModel):
    layer1_ncf: float
    layer2_cbf: float
    layer12_combined: float
    layer3_bayesian_raw: float
    layer3_bayesian_norm: float
    hybrid_score: float


class RecommendedItem(BaseModel):
    product_id: int
    product_name: str
    variant: str
    category: str
    price: float
    scores: ScoreDetail


class CartSummary(BaseModel):
    items_count: int
    total_value: float
    identified_products: list[dict]


class RecommendResponse(BaseModel):
    customer_segment: str
    cluster_id: int
    weights: dict[str, float]
    cart_summary: CartSummary
    recommendations: list[RecommendedItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness probe")
def health():
    return {
        "status": "ok",
        "model_loaded": _recommender is not None,
    }


@app.get("/products", summary="Daftar seluruh produk dalam katalog")
def list_products(
    category: Annotated[Optional[str], Query(description="Filter by category")] = None
):
    if _recommender is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat. Jalankan train.py terlebih dahulu.")

    df = _recommender.products[["ProductId", "ProductName", "Variant", "Category", "Price", "Description"]]
    if category:
        df = df[df["Category"].str.lower() == category.lower()]

    return {
        "total": len(df),
        "products": df.to_dict(orient="records"),
    }


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Dapatkan rekomendasi berdasarkan isi keranjang",
    description=(
        "Kirimkan daftar menu yang sedang ada di keranjang belanja. "
        "Sistem akan:\n"
        "1. Menentukan segmen pelanggan via RFM pseudo-profiling\n"
        "2. Menghasilkan kandidat via Neural CF (Layer 1)\n"
        "3. Menyempurnakan peringkat via Content-Based Filtering (Layer 2)\n"
        "4. Menyesuaikan dengan Bayesian Average rating (Layer 3)\n"
        "5. Menggabungkan skor akhir: **Hybrid = a·(NCF+CBF)/2 + b·BayesianNorm**"
    ),
)
def recommend(request: RecommendRequest):
    if _recommender is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat. Jalankan train.py terlebih dahulu.")

    cart_items = [item.model_dump() for item in request.cart]

    result = _recommender.recommend(
        cart_items=cart_items,
        top_n=request.top_n,
        exclude_cart=request.exclude_cart,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
