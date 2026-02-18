"""
Stage 0 — Static Dataset Loader
Loads pre-downloaded labeled datasets to supplement live-scraped data:

  1. Financial PhraseBank (Malo et al. 2014)
     - Source: HuggingFace / Kaggle
     - Path: data/raw/phrasebank_75agree.csv
     - Columns: text, label (0=negative, 1=neutral, 2=positive)

  2. Kaggle Stock News Sentiment (avisheksood)
     - Path: data/raw/stock_news_sentiment/ (unzipped)
     - Columns vary — auto-detected and normalised

Output:
    data/raw/static_phrasebank.csv
    data/raw/static_stocknews.csv
    data/raw/static_combined.csv   ← merged, normalised, ready for Stage 1B
"""

import pandas as pd
import logging
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RAW_DATA_DIR, TIMESTAMP_FORMAT, TIMEZONE
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
PHRASEBANK_PATH  = os.path.join(RAW_DATA_DIR, "phrasebank_75agree.csv")
STOCKNEWS_DIR    = os.path.join(RAW_DATA_DIR, "stock_news_sentiment")
STOCKNEWS_GLOB   = os.path.join(RAW_DATA_DIR, "*.csv")   # fallback: scan raw dir

# Label normalisation map (handles multiple naming conventions)
LABEL_MAP_TEXT = {
    "positive": 2, "pos": 2, "bullish": 2, "buy": 2, "1": 2,
    "neutral":  1, "neu": 1, "hold":    1, "0": 1,
    "negative": 0, "neg": 0, "bearish": 0, "sell": 0, "-1": 0,
}
LABEL_MAP_INT = {2: 2, 1: 1, 0: 0, -1: 0}


def _normalise_label(val) -> int | None:
    """Convert any label format to 0/1/2."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in LABEL_MAP_TEXT:
        return LABEL_MAP_TEXT[s]
    try:
        i = int(float(s))
        return LABEL_MAP_INT.get(i)
    except ValueError:
        return None


def load_phrasebank() -> pd.DataFrame:
    """
    Load Financial PhraseBank (75% agreement subset).
    Expects columns: text, label (0/1/2).
    If not present, run the extraction script first (see project README).
    """
    if not os.path.exists(PHRASEBANK_PATH):
        logger.warning(f"PhraseBank not found at {PHRASEBANK_PATH}. "
                       "Run the extraction script to generate it.")
        return pd.DataFrame()

    df = pd.read_csv(PHRASEBANK_PATH)
    logger.info(f"PhraseBank raw: {len(df)} rows")

    # Normalise columns
    df.columns = [c.lower().strip() for c in df.columns]
    if "text" not in df.columns or "label" not in df.columns:
        logger.error("PhraseBank CSV must have 'text' and 'label' columns.")
        return pd.DataFrame()

    df["label"] = df["label"].apply(_normalise_label)
    df.dropna(subset=["text", "label"], inplace=True)
    df["label"] = df["label"].astype(int)

    df["source"]       = "phrasebank"
    df["type"]         = "news_headline"
    df["market"]       = "US"
    df["timestamp_utc"] = None   # static dataset — no timestamps

    logger.info(f"PhraseBank loaded: {len(df)} rows | label dist: {df['label'].value_counts().to_dict()}")
    return df[["text", "label", "source", "type", "market", "timestamp_utc"]]


def _detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Auto-detect text and label column names from a DataFrame."""
    text_candidates  = ["text", "sentence", "news", "headline", "title", "message", "content"]
    label_candidates = ["label", "sentiment", "sentiment_label", "target", "class", "score"]

    cols_lower = {c.lower(): c for c in df.columns}

    text_col  = next((cols_lower[c] for c in text_candidates  if c in cols_lower), None)
    label_col = next((cols_lower[c] for c in label_candidates if c in cols_lower), None)

    return text_col, label_col


def load_stocknews_kaggle() -> pd.DataFrame:
    """
    Load Kaggle Stock News Sentiment dataset.
    Searches STOCKNEWS_DIR first, then falls back to scanning data/raw/ for CSVs
    that look like the Kaggle dataset.
    """
    # Find candidate CSV files
    candidates = []
    if os.path.isdir(STOCKNEWS_DIR):
        candidates = glob.glob(os.path.join(STOCKNEWS_DIR, "**", "*.csv"), recursive=True)
    if not candidates:
        # Fallback: look for any large CSV in raw dir (excluding ones we know)
        known = {"phrasebank_75agree.csv", "static_phrasebank.csv",
                 "static_stocknews.csv", "static_combined.csv"}
        candidates = [
            f for f in glob.glob(STOCKNEWS_GLOB)
            if os.path.basename(f) not in known
            and "reddit" not in os.path.basename(f)
            and "news_raw" not in os.path.basename(f)
        ]

    if not candidates:
        logger.warning("Kaggle Stock News CSV not found. "
                       "Download it with: kaggle datasets download "
                       "-d avisheksood/stock-news-sentiment-analysismassive-dataset "
                       "-p data/raw --unzip")
        return pd.DataFrame()

    frames = []
    for path in candidates:
        try:
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
            text_col, label_col = _detect_columns(df)
            if text_col is None or label_col is None:
                logger.warning(f"Could not detect text/label columns in {path} — skipping")
                continue

            df = df[[text_col, label_col]].copy()
            df.columns = ["text", "label"]
            df["label"] = df["label"].apply(_normalise_label)
            df.dropna(subset=["text", "label"], inplace=True)
            df["label"] = df["label"].astype(int)
            df["source"]        = "kaggle_stocknews"
            df["type"]          = "news_headline"
            df["market"]        = "US"
            df["timestamp_utc"] = None
            frames.append(df)
            logger.info(f"Loaded {len(df)} rows from {os.path.basename(path)}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset=["text"], inplace=True)
    logger.info(f"Kaggle StockNews total: {len(combined)} rows | "
                f"label dist: {combined['label'].value_counts().to_dict()}")
    return combined[["text", "label", "source", "type", "market", "timestamp_utc"]]


def run_static_dataset_loading() -> pd.DataFrame:
    """
    Entry point: load all static datasets, merge, save.
    Returns combined DataFrame with columns: text, label, source, type, market, timestamp_utc.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    phrasebank_df  = load_phrasebank()
    stocknews_df   = load_stocknews_kaggle()

    frames = [df for df in [phrasebank_df, stocknews_df] if not df.empty]

    if not frames:
        logger.error("No static datasets loaded. Check file paths.")
        return pd.DataFrame()

    # Save individual files
    if not phrasebank_df.empty:
        phrasebank_df.to_csv(os.path.join(RAW_DATA_DIR, "static_phrasebank.csv"), index=False)
        logger.info("Saved static_phrasebank.csv")

    if not stocknews_df.empty:
        stocknews_df.to_csv(os.path.join(RAW_DATA_DIR, "static_stocknews.csv"), index=False)
        logger.info("Saved static_stocknews.csv")

    logger.info(f"Static phrasebank: {len(phrasebank_df)} rows")
    logger.info(f"Static stocknews: {len(stocknews_df)} rows")

    # Return dict so callers can use each separately
    return {"phrasebank": phrasebank_df, "stocknews": stocknews_df}


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
    df = run_static_dataset_loading()
    if not df.empty:
        print(f"\nTotal static records: {len(df)}")
        print(df.groupby(["source", "label"]).size().unstack(fill_value=0))