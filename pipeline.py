"""
Predictive Markets Pipeline
----------------------------
Runs nightly via GitHub Actions:
 1. Fetches Polymarket events  -> poly_df
 2. Fetches Kalshi events      -> df_kalshi_filtered
 3. Semantic matching          -> matches
 4. Merged output              -> kalshi_poly_df

All outputs saved as .parquet AND .json inside ./data/
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

POLY_BASE_URL = "https://gamma-api.polymarket.com/events"
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
LIMIT = 100
DELAY = 0.5
MAX_RETRIES = 3
KALSHI_MAX_PAGES = 50   # 50 × 200 = 10 000 eventos max → evita runner timeout

# Columns to keep in poly_df when saving to disk (avoids 300MB+ files).
# The full object is still used in-memory for matching.
POLY_SLIM_COLS = [
    "id", "title", "description", "endDate", "startDate",
    "volume", "liquidity", "active", "closed", "category", "tags",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, name: str):
    """Save DataFrame as both .parquet and .json inside DATA_DIR."""
    parquet_path = os.path.join(DATA_DIR, f"{name}.parquet")
    json_path    = os.path.join(DATA_DIR, f"{name}.json")

    # Parquet: convert problematic object columns to string to avoid schema issues
    df_parquet = df.copy()
    for col in df_parquet.select_dtypes(include="object").columns:
        df_parquet[col] = df_parquet[col].apply(
            lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
        )
    df_parquet.to_parquet(parquet_path, index=False)

    # JSON: use records orientation for readability
    df.to_json(json_path, orient="records", indent=2, date_format="iso")

    print(f"  💾 {name}.parquet  ({os.path.getsize(parquet_path) / 1024 / 1024:.1f} MB)")
    print(f"  💾 {name}.json     ({os.path.getsize(json_path) / 1024 / 1024:.1f} MB)")


def save_slim(df: pd.DataFrame, name: str, keep_cols: list):
    """Save only a subset of columns to keep file size small for git."""
    available = [c for c in keep_cols if c in df.columns]
    save(df[available], name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. POLYMARKET
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_polymarket_page(offset: int) -> list | None:
    """
    Fetch one page from Polymarket.
    Returns the list of events, an empty list if the page is empty,
    or None if the API signals end-of-data (422) so the caller can stop.
    Retries only on transient errors (429, 5xx, timeouts, connection drops).
    """
    params = {
        "order": "id",
        "ascending": "false",
        "closed": "false",
        "limit": LIMIT,
        "offset": offset,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(POLY_BASE_URL, params=params, timeout=30)

            # 422 = offset beyond API limit → treat as clean end-of-data
            if resp.status_code == 422:
                print(f"  ℹ️  Polymarket: 422 at offset={offset}, stopping pagination.")
                return None

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as err:
            status = err.response.status_code if err.response is not None else None
            # Only retry transient server-side errors
            if status in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES - 1:
                print(f"  ⚠️  Polymarket attempt {attempt + 1}/{MAX_RETRIES} (HTTP {status}): retrying in 5s…")
                time.sleep(5)
                continue
            raise  # non-transient HTTP error → bubble up immediately

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            if attempt < MAX_RETRIES - 1:
                print(f"  ⚠️  Polymarket attempt {attempt + 1}/{MAX_RETRIES} ({err}): retrying in 5s…")
                time.sleep(5)
                continue
            raise

    return None  # exhausted retries


def fetch_polymarket() -> pd.DataFrame:
    print("\n📡 Fetching Polymarket data...")
    all_events: list = []
    offset = 0

    while True:
        page = _fetch_polymarket_page(offset)

        # None  → 422 end-of-data signal
        # []    → empty page → done
        if not page:
            break

        all_events.extend(page)

        # Partial page → last page
        if len(page) < LIMIT:
            break

        offset += LIMIT
        time.sleep(DELAY)

    print(f"  ✅ Polymarket: {len(all_events)} events fetched.")
    return pd.DataFrame(all_events)


# ─────────────────────────────────────────────────────────────────────────────
# 2. KALSHI
# ─────────────────────────────────────────────────────────────────────────────

def fetch_kalshi() -> pd.DataFrame:
    print("\n📡 Fetching Kalshi data...")
    all_events = []
    page_size = 200
    pages_fetched = 0

    def _get_with_retry(url: str) -> dict:
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as err:
                print(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES}: {err}. Retrying in 5s...")
                time.sleep(5)
        raise RuntimeError(f"Kalshi API failed after {MAX_RETRIES} retries: {url}")

    data = _get_with_retry(
        f"{KALSHI_BASE_URL}?limit={page_size}&with_nested_markets=true&status=open"
    )
    all_events.extend(data.get("events", []))
    pages_fetched += 1

    while data.get("cursor") and pages_fetched < KALSHI_MAX_PAGES:
        time.sleep(DELAY)
        data = _get_with_retry(
            f"{KALSHI_BASE_URL}?cursor={data['cursor']}&limit={page_size}&with_nested_markets=true&status=open"
        )
        all_events.extend(data.get("events", []))
        pages_fetched += 1

    if pages_fetched >= KALSHI_MAX_PAGES:
        print(f"  ⚠️  Kalshi: page limit ({KALSHI_MAX_PAGES}) reached — truncating fetch.")

    print(f"  ✅ Kalshi: {len(all_events)} events fetched ({pages_fetched} pages).")


    # Flatten nested markets
    data_rows = []
    for event in all_events:
        title    = event.get("title", "")
        ticker   = event.get("event_ticker", "")
        category = event.get("category", "")
        markets  = event.get("markets") or []

        for market in markets:
            data_rows.append({
                "title":                    title,
                "rules_primary":            market.get("rules_primary"),
                "category":                 category,
                "status":                   market.get("status"),
                "expected_expiration_time": market.get("expected_expiration_time"),
                "event_ticker":             ticker,
                "event_ticker2":            market.get("ticker"),
                "yes_sub_title":            market.get("yes_sub_title"),
                "yes_bid_dollars":          market.get("yes_bid_dollars"),
                "yes_ask_dollars":          market.get("yes_ask_dollars"),
                "no_bid_dollars":           market.get("no_bid_dollars"),
                "no_ask_dollars":           market.get("no_ask_dollars"),
                "expiration_time":          market.get("expiration_time"),
                "volume":                   market.get("volume", 0),
            })

    df = pd.DataFrame(data_rows)
    if not df.empty:
        df["expiration_time"]          = pd.to_datetime(df["expiration_time"],          errors="coerce")
        df["expected_expiration_time"] = pd.to_datetime(df["expected_expiration_time"], errors="coerce")
        df = df[df["status"] == "active"].reset_index(drop=True)

    print(f"  ✅ Kalshi active markets: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEMANTIC MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def match_markets(poly_df: pd.DataFrame, kalshi: pd.DataFrame) -> pd.DataFrame:
    print("\n🧠 Running semantic matching...")

    kalshi = kalshi.copy()
    polymarket = poly_df.copy()

    kalshi["bet_description"] = (
        kalshi["title"].fillna("") + " " +
        kalshi["rules_primary"].fillna("")
    )
    polymarket["bet_description"] = (
        polymarket["title"].fillna("") + " " +
        polymarket.get("description", pd.Series(dtype=str)).fillna("")
    )

    kalshi    = kalshi.dropna(subset=["bet_description"])
    polymarket = polymarket.dropna(subset=["bet_description"])

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  Encoding Polymarket embeddings...")
    poly_emb   = model.encode(polymarket["bet_description"].tolist(), show_progress_bar=True)
    print("  Encoding Kalshi embeddings...")
    kalshi_emb = model.encode(kalshi["bet_description"].tolist(),     show_progress_bar=True)

    similarity = model.similarity(kalshi_emb, poly_emb)
    top_match  = similarity.argmax(axis=1)

    matches = pd.DataFrame({
        "kalshi_title":       kalshi["title"].values,
        "matched_polymarket": polymarket.iloc[top_match]["title"].values,
    })
    print(f"  ✅ {len(matches)} matches generated.")

    # ── Merge with Polymarket market details ─────────────────────────────────
    merged = matches.merge(
        polymarket[["title", "endDate", "markets"]],
        left_on="matched_polymarket",
        right_on="title",
        how="left",
    )

    market_data = []
    for _, row in merged.iterrows():
        markets_list = row["markets"]
        if markets_list is not None and isinstance(markets_list, list) and len(markets_list) > 0:
            mkt = markets_list[0]
            market_data.append({
                "kalshi_title":        row["kalshi_title"],
                "matched_polymarket":  row["matched_polymarket"],
                "poly_endDate":        row.get("endDate"),
                "poly_question":       mkt.get("question"),
                "poly_outcomePrices":  mkt.get("outcomePrices"),
                "poly_lastTradePrice": mkt.get("lastTradePrice"),
                "poly_bestBid":        mkt.get("bestBid"),
                "poly_bestAsk":        mkt.get("bestAsk"),
                "poly_volume":         mkt.get("volume"),
                "poly_market_id":      mkt.get("id"),
            })

    result_df = pd.DataFrame(market_data)

    # ── Merge with Kalshi pricing data ────────────────────────────────────────
    kalshi_merge = (
        kalshi[[
            "title", "category", "event_ticker",
            "expiration_time", "yes_bid_dollars", "yes_ask_dollars",
            "no_bid_dollars", "no_ask_dollars",
        ]]
        .rename(columns={
            "title":            "kalshi_title",
            "category":         "kalshi_category",
            "event_ticker":     "kalshi_market_id",
            "expiration_time":  "kalshi_expiration_time",
            "yes_bid_dollars":  "kalshi_yes_bid_dollars",
            "yes_ask_dollars":  "kalshi_yes_ask_dollars",
            "no_bid_dollars":   "kalshi_no_bid_dollars",
            "no_ask_dollars":   "kalshi_no_ask_dollars",
        })
        .drop_duplicates(subset=["kalshi_title"])
    )

    kalshi_poly_df = result_df.merge(kalshi_merge, on="kalshi_title", how="left")
    print(f"  ✅ kalshi_poly_df shape: {kalshi_poly_df.shape}")
    return kalshi_poly_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PREDICTIVE MARKETS PIPELINE")
    print("=" * 60)

    # 1. Fetch data
    try:
        poly_df = fetch_polymarket()
    except Exception as e:
        raise RuntimeError(f"🛑 Polymarket fetch failed, aborting pipeline: {e}") from e

    if poly_df.empty:
        raise RuntimeError("🛑 Polymarket returned 0 events — aborting pipeline.")

    try:
        df_kalshi_filtered = fetch_kalshi()
    except Exception as e:
        raise RuntimeError(f"🛑 Kalshi fetch failed, aborting pipeline: {e}") from e

    # 2. Save raw data
    print("\n💾 Saving raw data...")
    # poly_df can be 300+ MB with the nested 'markets' column — save a slim version.
    # The full DataFrame is still used in-memory for matching.
    save_slim(poly_df, "poly_df", POLY_SLIM_COLS)
    save(df_kalshi_filtered, "df_kalshi_filtered")

    # 3. Match markets
    kalshi_poly_df = match_markets(poly_df, df_kalshi_filtered)

    # 4. Save final output
    print("\n💾 Saving matched data...")
    save(kalshi_poly_df, "kalshi_poly_df")

    print("\n✅ Pipeline completed successfully!")
    print(f"   poly_df:            {poly_df.shape}")
    print(f"   df_kalshi_filtered: {df_kalshi_filtered.shape}")
    print(f"   kalshi_poly_df:     {kalshi_poly_df.shape}")


if __name__ == "__main__":
    main()
