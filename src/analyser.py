"""
STAGE 1C — SENTIMENT FEATURE AGGREGATION
==========================================
Aggregates post-level sentiment predictions (from Stage 1B / finbert.py)
into hourly features per ticker.

Input:  CSV from finbert.py with sentiment predictions per post
Output: Aggregated features per (ticker, hour) with LAG applied

Features computed (per ticker per window):
  1. avg_sentiment         — Credibility-weighted mean sentiment score
  2. sentiment_std         — Standard deviation (volatility of crowd opinion)
  3. pos_count             — Number of positive posts
  4. neg_count             — Number of negative posts
  5. neu_count             — Number of neutral posts
  6. bull_bear_ratio       — pos / (pos + neg), range 0-1
  7. mention_volume        — Total post count
  8. weighted_volume       — Sum of credibility weights (better volume measure)
  9. sentiment_momentum    — Change from previous window
  10. sentiment_acceleration — Change in momentum
  11. high_confidence_ratio — % of posts with confidence > 0.80

CRITICAL: Features are LAG-SHIFTED by 1 window to prevent data leakage.
Sentiment at 13:00 is joined to price at 14:00.

Usage:
  from analyser import aggregate_sentiment_features
  features_df = aggregate_sentiment_features(sentiment_csv)
"""

import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Feature computation (per group)
# ─────────────────────────────────────────────

def _safe_bull_bear(pos: int, neg: int) -> float:
    """pos / (pos + neg); returns 0.5 when both are zero (neutral stance)."""
    denom = pos + neg
    return pos / denom if denom > 0 else 0.5


def _compute_window_features(group: pd.DataFrame) -> pd.Series:
    """
    Called once per (ticker, window) group.
    Expects columns: weighted_sentiment_score, sentiment_label,
                     sentiment_confidence, user_credibility
    """
    scores      = group["weighted_sentiment_score"]
    labels      = group["sentiment_label"]
    creds       = group["user_credibility"].fillna(1.0)
    confidences = group["sentiment_confidence"]

    pos_mask = labels == "positive"
    neg_mask = labels == "negative"
    neu_mask = labels == "neutral"

    pos_count = pos_mask.sum()
    neg_count = neg_mask.sum()
    neu_count = neu_mask.sum()

    # Weighted mean (credibility as weights)
    total_cred = creds.sum()
    avg_sentiment = (scores * creds).sum() / total_cred if total_cred > 0 else 0.0

    return pd.Series({
        "avg_sentiment":         avg_sentiment,
        "sentiment_std":         scores.std(ddof=0) if len(scores) > 1 else 0.0,
        "pos_count":             int(pos_count),
        "neg_count":             int(neg_count),
        "neu_count":             int(neu_count),
        "bull_bear_ratio":       _safe_bull_bear(pos_count, neg_count),
        "mention_volume":        len(group),
        "weighted_volume":       float(creds.sum()),
        "high_confidence_ratio": float((confidences > 0.80).mean()),
    })


# ─────────────────────────────────────────────
# Momentum & acceleration (per ticker time-series)
# ─────────────────────────────────────────────

def _add_momentum_features(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """
    ticker_df is already sorted by window_start (ascending) for one ticker.
    Adds sentiment_momentum and sentiment_acceleration in-place.
    """
    avg = ticker_df["avg_sentiment"]
    ticker_df["sentiment_momentum"]     = avg.diff(1)          # t − (t−1)
    ticker_df["sentiment_acceleration"] = ticker_df["sentiment_momentum"].diff(1)
    return ticker_df


# ─────────────────────────────────────────────
# Main aggregation function
# ─────────────────────────────────────────────

def aggregate_sentiment_features(
    posts_df: Union[pd.DataFrame, str, Path],
    timestamp_col:   Optional[str] = None,
    ticker_col:      str = "ticker",
    freq:            str = "1h",
    lag_windows:     int = 1,
) -> pd.DataFrame:
    """
    Aggregate sentiment predictions into hourly features per ticker.
    
    Parameters
    ----------
    posts_df       : Either:
                     - DataFrame from finbert.py (Stage 1B output)
                     - Path to *_sentiment.csv file
                     Must have columns: ticker, timestamp, sentiment_label,
                     sentiment_numeric, sentiment_confidence, 
                     weighted_sentiment_score, user_credibility
    timestamp_col  : Name of timestamp column. If None, auto-detects from:
                     ['timestamp_utc', 'post_timestamp', 'timestamp']
    ticker_col     : Column holding ticker symbols (default: 'ticker')
    freq           : Aggregation window (default: '1h' for hourly)
    lag_windows    : How many windows to lag (default: 1)
                     1 = sentiment at 13:00 joins to price at 14:00

    Returns
    -------
    DataFrame indexed by (ticker, window_start) with all 11 features.
    window_start has already been shifted forward by `lag_windows` so you can
    join directly to price data on the timestamp without any further shifting.
    The column `feature_source_window` records the *original* window the
    sentiment was computed from (for auditing).
    """

    # ── Load data if path provided ──────────────────────────────────
    if isinstance(posts_df, (str, Path)):
        logger.info(f"Loading sentiment predictions from: {posts_df}")
        df = pd.read_csv(posts_df)
    else:
        df = posts_df.copy()
    
    # ── Validate required columns ────────────────────────────────────
    required = ['sentiment_label', 'sentiment_numeric', 'sentiment_confidence', 
                'weighted_sentiment_score']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns from Stage 1B: {missing}\n"
            f"Expected output from finbert.py with sentiment predictions."
        )
    
    # ── Auto-detect timestamp column ─────────────────────────────────
    if timestamp_col is None:
        candidates = ['timestamp_utc', 'post_timestamp', 'timestamp']
        timestamp_col = next((c for c in candidates if c in df.columns), None)
        if timestamp_col is None:
            raise ValueError(
                f"No timestamp column found. Expected one of: {candidates}\n"
                f"Available columns: {list(df.columns)}"
            )
        logger.info(f"Using timestamp column: {timestamp_col}")
    
    # ── Validate ticker column ───────────────────────────────────────
    if ticker_col not in df.columns:
        raise ValueError(
            f"Ticker column '{ticker_col}' not found.\n"
            f"Stage 1A (NER) must add ticker attribution before Stage 1B.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # ── Add user_credibility if missing ──────────────────────────────
    if 'user_credibility' not in df.columns:
        logger.warning("No 'user_credibility' column found. Defaulting all to 1.0")
        df['user_credibility'] = 1.0
    
    # ── Ensure datetime ──────────────────────────────────────────────
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    
    # ── Filter out invalid data ──────────────────────────────────────
    initial_len = len(df)
    df = df.dropna(subset=[timestamp_col, ticker_col, 'weighted_sentiment_score'])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with missing data")

    # ── Bucket each post into its time window ────────────────────────
    df["window_start"] = df[timestamp_col].dt.floor(freq)

    # ── Group and aggregate ──────────────────────────────────────────
    grouped = (
        df.groupby([ticker_col, "window_start"], observed=True)
          .apply(_compute_window_features)
          .reset_index()
    )
    grouped.rename(columns={ticker_col: "ticker"}, inplace=True)
    grouped.sort_values(["ticker", "window_start"], inplace=True)

    # ── Momentum & acceleration (per-ticker) ─────────────────────────
    grouped = (
        grouped.groupby("ticker", group_keys=False)
               .apply(_add_momentum_features)
    )

    # ── LAG ENFORCEMENT (non-negotiable) ─────────────────────────────
    # Record what window these features actually describe…
    grouped["feature_source_window"] = grouped["window_start"]
    # …then shift the index forward so rows align with the NEXT price bar.
    freq_offset = pd.tseries.frequencies.to_offset(freq)
    grouped["window_start"] = (
        grouped["window_start"] + lag_windows * freq_offset
    )

    logger.info(
        f"Stage 1C complete.  "
        f"Tickers: {grouped['ticker'].nunique()}  |  "
        f"Windows: {grouped['window_start'].nunique()}  |  "
        f"Total feature rows: {len(grouped)}  |  "
        f"Lag applied: {lag_windows} x {freq}"
    )
    
    # ── Summary statistics ───────────────────────────────────────────
    if logger.level <= logging.INFO:
        print("\n" + "="*70)
        print("  STAGE 1C SUMMARY")
        print("="*70)
        print(f"Tickers processed: {', '.join(sorted(grouped['ticker'].unique()))}")
        print(f"Time range: {grouped['window_start'].min()} to {grouped['window_start'].max()}")
        print(f"\nAverage sentiment by ticker:")
        ticker_summary = grouped.groupby('ticker')['avg_sentiment'].agg(['count', 'mean', 'std'])
        ticker_summary.columns = ['Windows', 'Avg_Sentiment', 'Std_Sentiment']
        print(ticker_summary.round(3).to_string())
        print("="*70 + "\n")

    return grouped.reset_index(drop=True)


# ─────────────────────────────────────────────
# STAGE 2C: Feature Fusion
# ─────────────────────────────────────────────

def join_to_price(
    price_df: pd.DataFrame,
    features_df: pd.DataFrame,
    price_time_col: str = "timestamp",
    price_ticker_col: str = "ticker",
    fill_missing: str = "zero",
) -> pd.DataFrame:
    """
    Left-joins the lagged sentiment features onto OHLCV price bars.
    This is Stage 2C in the pipeline: combining sentiment + market features.

    Parameters
    ----------
    price_df         : OHLCV data with timestamp and ticker columns
    features_df      : Output from aggregate_sentiment_features()
    price_time_col   : Timestamp column in price_df (default: 'timestamp')
    price_ticker_col : Ticker column in price_df (default: 'ticker')
    fill_missing     : How to handle missing sentiment:
                       'zero' = fill with 0 (no sentiment that hour)
                       'ffill' = forward-fill last known sentiment
                       'drop' = remove rows without sentiment
                       'keep' = leave as NaN
    
    Returns
    -------
    DataFrame with price bars + sentiment features merged.
    windows without sentiment data are handled per fill_missing strategy.
    """
    price_df = price_df.copy()
    price_df[price_time_col] = pd.to_datetime(price_df[price_time_col], utc=True)

    merged = price_df.merge(
        features_df,
        left_on  =[price_ticker_col, price_time_col],
        right_on =["ticker",         "window_start"],
        how      ="left",
        suffixes =('', '_sentiment')
    )

    dup_ticker = price_ticker_col != "ticker"
    if dup_ticker:
        merged.drop(columns=["ticker"], inplace=True, errors="ignore")
    
    # Clean up duplicate columns from merge
    # Drop timestamp_utc_sentiment if it exists (we keep the price timestamp)
    merged.drop(columns=["timestamp_utc_sentiment"], inplace=True, errors="ignore")
    # Drop window_start after merge (we have timestamp_utc)
    merged.drop(columns=["window_start"], inplace=True, errors="ignore")

    # ── Handle missing sentiment data ────────────────────────────────
    sentiment_cols = ['avg_sentiment', 'sentiment_std', 'pos_count', 'neg_count',
                     'neu_count', 'bull_bear_ratio', 'mention_volume', 
                     'weighted_volume', 'sentiment_momentum', 
                     'sentiment_acceleration', 'high_confidence_ratio']
    
    matched_count = merged['avg_sentiment'].notna().sum()
    missing_count = len(merged) - matched_count

    # Track whether sentiment existed at the exact aligned timestamp before imputation.
    merged['sentiment_available'] = merged['avg_sentiment'].notna().astype(int)
    pre_fill_missing = merged['avg_sentiment'].isna()
    
    if fill_missing == "zero":
        merged[sentiment_cols] = merged[sentiment_cols].fillna(0)
        merged['sentiment_imputed'] = pre_fill_missing.astype(int)
        logger.info(f"Filled {missing_count} rows with zero sentiment (no posts that hour)")
    elif fill_missing == "ffill":
        ffill_limit = int(os.getenv("SENTIMENT_FFILL_LIMIT", "24"))
        merged[sentiment_cols] = merged.groupby(price_ticker_col)[sentiment_cols].ffill(limit=ffill_limit)
        merged['sentiment_imputed'] = (pre_fill_missing & merged['avg_sentiment'].notna()).astype(int)
        logger.info(
            f"Forward-filled {missing_count} rows with last known sentiment "
            f"(limit={ffill_limit} bars)"
        )
    elif fill_missing == "drop":
        merged = merged.dropna(subset=['avg_sentiment'])
        merged['sentiment_imputed'] = 0
        logger.info(f"Dropped {missing_count} rows without sentiment data")
    else:
        # keep NaN path
        merged['sentiment_imputed'] = 0
    # elif fill_missing == "keep": do nothing
    
    logger.info(
        f"Feature fusion complete.  "
        f"Price rows: {len(price_df)}  |  "
        f"Final rows: {len(merged)}  |  "
        f"Matched with sentiment: {matched_count}"
    )
    return merged


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # Simulate post-level output from Stage 1B
    n = 200
    tickers = np.random.choice(["TSLA", "AAPL", "NVDA"], n)
    times   = pd.date_range("2024-01-15 09:00", periods=n, freq="3min", tz="UTC")
    labels  = np.random.choice(["positive", "negative", "neutral"], n, p=[0.45, 0.30, 0.25])
    num_map = {"positive": 1, "neutral": 0, "negative": -1}

    posts = pd.DataFrame({
        "post_timestamp":          times,
        "ticker":                  tickers,
        "sentiment_label":         labels,
        "sentiment_numeric":       [num_map[l] for l in labels],
        "sentiment_confidence":    np.random.uniform(0.50, 0.99, n),
        "user_credibility":        np.random.uniform(0.3, 1.0, n),
        "is_uncertain":            np.random.choice([True, False], n, p=[0.15, 0.85]),
    })
    posts["weighted_sentiment_score"] = (
        posts["sentiment_numeric"]
        * posts["sentiment_confidence"]
        * posts["user_credibility"]
        * np.where(posts["is_uncertain"], 0.30, 1.0)
    )

    features = aggregate_sentiment_features(posts)
    print("\n── Feature table (first 12 rows) ──")
    print(features.head(12).to_string(index=False))

    # Simulate price bars and join
    price_times = pd.date_range("2024-01-15 10:00", periods=6, freq="1h", tz="UTC")
    price_df = pd.DataFrame({
        "bar_timestamp": list(price_times) * 3,
        "ticker":        ["TSLA"]*6 + ["AAPL"]*6 + ["NVDA"]*6,
        "close":         np.random.uniform(100, 500, 18),
    })

    merged = join_to_price(price_df, features)
    print("\n── Price + sentiment (first 10 rows) ──")
    print(merged[["ticker","bar_timestamp","close","avg_sentiment",
                  "bull_bear_ratio","mention_volume","sentiment_momentum"]].head(10).to_string(index=False))