"""Data loading and preprocessing utilities shared across all layers."""
import re
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


def load_products() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "products.csv")
    df["ProductId"] = pd.to_numeric(df["ProductId"], errors="coerce").astype("Int64")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
    df["Variant"] = df["Variant"].fillna("-").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["Category"] = df["Category"].fillna("").astype(str)
    df["ProductName"] = df["ProductName"].fillna("").astype(str)
    return df.dropna(subset=["ProductId"]).reset_index(drop=True)


def load_reviews() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "reviews.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Menu"] = pd.to_numeric(df["Menu"], errors="coerce").astype("Int64")
    return df.dropna(subset=["Rating", "Menu"]).reset_index(drop=True)


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "transactions.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce")
    df["TotalItem"] = pd.to_numeric(df["TotalItem"], errors="coerce")
    return df.dropna(subset=["Date", "TotalAmount", "TotalItem", "Items"]).reset_index(
        drop=True
    )


def parse_item_string(item_str: str) -> tuple[str, str]:
    """Split 'ProductName (Variant)' into (name, variant). Returns (name, '-') if no variant."""
    item_str = item_str.strip()
    match = re.match(r"^(.+?)\s*\((.+?)\)\s*$", item_str)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return item_str, "-"


def build_product_lookup(products: pd.DataFrame) -> dict[str, int]:
    """
    Build a case-insensitive lookup from item string → ProductId.
    Priority order: exact 'Name (Variant)' → exact 'Name' → prefix match.
    """
    lookup: dict[str, int] = {}
    for _, row in products.iterrows():
        pid = int(row["ProductId"])
        name = str(row["ProductName"]).strip()
        variant = str(row["Variant"]).strip()

        if variant and variant != "-":
            lookup[f"{name} ({variant})".lower()] = pid

        # Name-only entry: only set if not already present (preserves first variant found)
        name_key = name.lower()
        if name_key not in lookup:
            lookup[name_key] = pid

    return lookup


def resolve_item_id(
    item_str: str, lookup: dict[str, int], products: pd.DataFrame
) -> int | None:
    """
    Map a raw item string (from transactions or API request) to a ProductId.
    Tries exact match first, then name-only, then substring fallback.
    """
    name, variant = parse_item_string(item_str)

    # 1. Exact full match
    if variant and variant != "-":
        key = f"{name} ({variant})".lower()
        if key in lookup:
            return lookup[key]

    # 2. Name-only match
    if name.lower() in lookup:
        return lookup[name.lower()]

    # 3. Substring fallback (product name contained within item_str)
    item_lower = item_str.lower()
    for k, pid in lookup.items():
        if k in item_lower:
            return pid

    return None


def expand_transactions(
    transactions: pd.DataFrame, lookup: dict[str, int], products: pd.DataFrame
) -> pd.DataFrame:
    """
    Expand each transaction row into one row per identified item.
    Returns DataFrame with columns: session_idx, product_id, date, total_amount, total_item.
    """
    records = []
    for sess_idx, row in transactions.iterrows():
        raw_items = str(row["Items"])
        items = [i.strip() for i in raw_items.split(",") if i.strip()]
        for item_str in items:
            pid = resolve_item_id(item_str, lookup, products)
            if pid is not None:
                records.append(
                    {
                        "session_idx": sess_idx,
                        "product_id": pid,
                        "date": row["Date"],
                        "total_amount": float(row["TotalAmount"]),
                        "total_item": int(row["TotalItem"]),
                    }
                )

    return pd.DataFrame(records)
