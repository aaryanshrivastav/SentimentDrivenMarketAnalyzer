"""
Stage 1A — Text Cleaner
Cleans raw CSVs from data/raw/ and saves cleaned versions to data/processed/.

Cleaning steps:
  - Decode HTML entities (&gt; &amp; etc.)
  - Strip HTML tags
  - Strip URLs
  - Strip ticker cashtags ($TSLA → TSLA removed)
  - Strip special characters
  - Collapse whitespace
  - Filter bot authors (VisualMod, AutoModerator) for Reddit files
  - Drop rows where cleaned text is under 20 chars

Usage:
    python src/text_cleaner.py
"""

import html
import pandas as pd
import re
import logging
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RAW_DATA_DIR, PROCESSED_DIR

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BOT_AUTHORS = ["VisualMod", "AutoModerator"]


# ── Cleaning functions ────────────────────────────────────────────────────────

def strip_html_entities(text: str) -> str:
    return html.unescape(text)

def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)

def strip_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", "", text)

def strip_cashtags(text: str) -> str:
    """Remove $TSLA style cashtags."""
    return re.sub(r"\$[A-Z]{1,5}\b", "", text)

def strip_special_chars(text: str) -> str:
    """Keep letters, numbers, and basic punctuation only."""
    return re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", " ", text)

def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = strip_html_entities(text)
    text = strip_html_tags(text)
    text = strip_urls(text)
    text = strip_cashtags(text)
    text = strip_special_chars(text)
    text = collapse_whitespace(text)
    return text


# ── File processing ───────────────────────────────────────────────────────────

def clean_file(input_path: str, output_path: str) -> int:
    df = pd.read_csv(input_path, low_memory=False)

    # Filter bot authors if this file has an author column (Reddit files)
    cols = df.columns.tolist()
    if "author" in cols:
        before = len(df)
        df = df[~df["author"].isin(BOT_AUTHORS)].copy()
        removed = before - len(df)
        if removed:
            logger.info(f"  Removed {removed} bot rows")

    # Detect text column
    text_col = next(
        (c for c in cols if c.lower() in ["text", "sentence", "headline", "content", "message"]),
        None
    )
    if text_col is None:
        logger.warning(f"  No text column found in {os.path.basename(input_path)} — skipping")
        return 0

    df["text_clean"] = df[text_col].apply(clean_text)

    # Drop rows that are too short after cleaning
    before = len(df)
    df = df[df["text_clean"].str.len() >= 20].copy()
    dropped = before - len(df)
    if dropped:
        logger.info(f"  Dropped {dropped} short/empty rows")

    df.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(df)} rows → {os.path.basename(output_path)}")
    return len(df)


def run_text_cleaning():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    patterns = [
        os.path.join(RAW_DATA_DIR, "reddit_raw_*.csv"),
        os.path.join(RAW_DATA_DIR, "news_raw_*.csv"),
        os.path.join(RAW_DATA_DIR, "static_phrasebank.csv"),
        os.path.join(RAW_DATA_DIR, "static_stocknews.csv"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    if not files:
        logger.warning("No raw files found to clean.")
        return

    logger.info(f"Found {len(files)} files to clean")
    total = 0

    for input_path in files:
        filename = os.path.basename(input_path)
        out_name = (filename
                    .replace("reddit_raw_", "reddit_clean_")
                    .replace("news_raw_", "news_clean_")
                    .replace("static_", "clean_"))
        output_path = os.path.join(PROCESSED_DIR, out_name)
        logger.info(f"Cleaning: {filename}")
        count = clean_file(input_path, output_path)
        total += count

    logger.info(f"Done. Total rows saved to processed/: {total}")


if __name__ == "__main__":
    run_text_cleaning()