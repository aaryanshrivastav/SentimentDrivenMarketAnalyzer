"""
Stage 0 — StockTwits Data Collector
Hits the public StockTwits stream endpoint (no auth required for basic pulls).
Pulls the last 30 messages per ticker (API hard cap without a paid key).

Output schema:
    id, source, ticker, author, text, upvotes (likes),
    account_followers, sentiment_label (StockTwits native if present),
    timestamp_utc, pull_timestamp_utc
"""

import requests
import pandas as pd
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    TICKERS, STOCKTWITS_BASE_URL, STOCKTWITS_LIMIT,
    RAW_DATA_DIR, TIMESTAMP_FORMAT, TIMEZONE
)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Polite delay between requests (StockTwits rate-limits aggressively)
REQUEST_DELAY_SECONDS = 2.0


def fetch_ticker_stream(ticker: str, session: requests.Session) -> list[dict]:
    """
    Fetch latest StockTwits messages for one ticker.
    Returns a list of record dicts.
    """
    url = STOCKTWITS_BASE_URL.format(ticker=ticker)
    pull_time = datetime.now(tz=TIMEZONE).strftime(TIMESTAMP_FORMAT)

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"StockTwits request failed for {ticker}: {e}")
        return []
    except ValueError as e:
        logger.error(f"JSON decode error for {ticker}: {e}")
        return []

    messages = data.get("messages", [])
    records = []

    for msg in messages[:STOCKTWITS_LIMIT]:
        # Extract native sentiment tag if the user set one
        sentiment_label = None
        if msg.get("entities", {}).get("sentiment"):
            sentiment_label = msg["entities"]["sentiment"].get("basic")  # "Bullish" / "Bearish"

        # Parse created_at timestamp → UTC
        created_at_str = msg.get("created_at", "")
        try:
            ts = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
            ts = ts.replace(tzinfo=TIMEZONE)
            timestamp_utc = ts.strftime(TIMESTAMP_FORMAT)
        except Exception:
            timestamp_utc = None

        user = msg.get("user", {})

        record = {
            "id":                 str(msg.get("id")),
            "source":             "stocktwits",
            "ticker":             ticker,
            "author":             user.get("username", ""),
            "text":               msg.get("body", ""),
            "upvotes":            msg.get("likes", {}).get("total", 0),
            "account_followers":  user.get("followers", 0),
            "account_following":  user.get("following", 0),
            "sentiment_label":    sentiment_label,     # native tag, useful signal
            "timestamp_utc":      timestamp_utc,
            "pull_timestamp_utc": pull_time,
        }
        records.append(record)

    logger.info(f"StockTwits {ticker}: {len(records)} messages fetched")
    return records


def run_stocktwits_collection(tickers: list[str] = None) -> pd.DataFrame:
    """
    Entry point: collect all tickers, save raw CSV.
    """
    if tickers is None:
        tickers = TICKERS

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    all_records = []
    for ticker in tickers:
        records = fetch_ticker_stream(ticker, session)
        all_records.extend(records)
        time.sleep(REQUEST_DELAY_SECONDS)   # be polite

    df = pd.DataFrame(all_records)

    if df.empty:
        logger.warning("No StockTwits data collected.")
        return df

    # Deduplicate
    before = len(df)
    df.drop_duplicates(subset=["id"], inplace=True)
    logger.info(f"Deduplicated: {before} → {len(df)} StockTwits records")

    # Save raw
    timestamp_tag = datetime.now(tz=TIMEZONE).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RAW_DATA_DIR, f"stocktwits_raw_{timestamp_tag}.csv")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved raw StockTwits data → {out_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
    df = run_stocktwits_collection()
    print(df.head())
    print(f"\nTotal records: {len(df)}")