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

    print(f"  💾 {name}.parquet  ({os.path.getsize(parquet_path) // 1024} KB)")
    print(f"  💾 {name}.json     ({os.path.getsize(json_path) // 1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# 1. POLYMARKET
# ─────────────────────────────────────────────────────────────────────────────

def fetch_polymarket() -> pd.DataFrame:
    print("\n📡 Fetching Polymarket data...")
    all_events = []
    offset = 0
    params = {
        "order": "id",
        "ascending": "false",
        "closed": "false",
        "limit": LIMIT,
    }

    while True:
        params["offset"] = offset
        events = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(POLY_BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                events = resp.json()
                if not events:
                    print(f"  ✅ Polymarket: {len(all_events)} events fetched.")
                    return pd.DataFrame(all_events)
                all_events.extend(events)
                offset += LIMIT
                time.sleep(DELAY)
                break
            except requests.exceptions.RequestException as err:
                print(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES}: {err}. Retrying in 5s...")
                time.sleep(5)

        if attempt == MAX_RETRIES - 1 and events is None:
            print("  🛑 Polymarket: giving up after retries.")
            break

    return pd.DataFrame(all_events)


# ─────────────────────────────────────────────────────────────────────────────
# 2. KALSHI
# ─────────────────────────────────────────────────────────────────────────────

def fetch_kalshi() -> pd.DataFrame:
    print("\n📡 Fetching Kalshi data...")
    all_events = []
    page_size = 200

    resp = requests.get(
        f"{KALSHI_BASE_URL}?limit={page_size}&with_nested_markets=true",
        timeout=60
    )
    data = resp.json()
    all_events.extend(data.get("events", []))

    while data.get("cursor"):
        resp = requests.get(
            f"{KALSHI_BASE_URL}?cursor={data['cursor']}&limit={page_size}&with_nested_markets=true",
            timeout=60
        )
        data = resp.json()
        all_events.extend(data.get("events", []))

    print(f"  ✅ Kalshi: {len(all_events)} events fetched.")

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
    poly_df           = fetch_polymarket()
    df_kalshi_filtered = fetch_kalshi()

    # 2. Save raw data
    print("\n💾 Saving raw data...")
    save(poly_df,            "poly_df")
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
